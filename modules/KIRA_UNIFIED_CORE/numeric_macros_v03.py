"""Exact rational, dimensionless, cost-gated periodic delta macros.

Macros describe repeating delta sequences, not geometric relations. Training
is per series; persistence is SQLite. Savings are computed in an explicit
symbol-count model, not claimed to be actual byte/time savings.
"""
import sqlite3
from fractions import Fraction
from numeric_core import rational


def encode(x):
    f = rational(x)
    return f'{f.numerator}/{f.denominator}'


def decode(x):
    a,b=x.split('/')
    return Fraction(int(a),int(b))


class NumericMacroLearner:
    def __init__(self, path=':memory:', max_period=16, min_savings=1):
        if max_period < 1: raise ValueError('max_period must be positive')
        self.max_period=max_period
        self.min_savings=min_savings
        self.db=sqlite3.connect(path)
        self.db.execute('CREATE TABLE IF NOT EXISTS macro_observations (series TEXT NOT NULL, seq INTEGER NOT NULL, val TEXT NOT NULL, PRIMARY KEY(series,seq))')
        self.db.execute('CREATE TABLE IF NOT EXISTS macro_rules (series TEXT PRIMARY KEY, period INTEGER NOT NULL, deltas TEXT NOT NULL, observations INTEGER NOT NULL, savings INTEGER NOT NULL)')
        self.db.commit()

    def observe(self, series, value):
        v=encode(value)
        with self.db:
            row=self.db.execute('SELECT COALESCE(MAX(seq),-1)+1 FROM macro_observations WHERE series=?',(series,)).fetchone()
            self.db.execute('INSERT INTO macro_observations VALUES (?,?,?)',(series,row[0],v))
        # No full-history scan on each observation. Consolidation is explicit.
        return None

    def consolidate(self, series):
        """Single O(m * max_period) scan; call at checkpoints, not per event."""
        with self.db:
            self._rebuild(series)
        return self.rule(series)

    def _values(self, series):
        return [decode(r[0]) for r in self.db.execute('SELECT val FROM macro_observations WHERE series=? ORDER BY seq',(series,))]

    def _rebuild(self, series):
        vals=self._values(series)
        ds=[b-a for a,b in zip(vals,vals[1:])]
        self.db.execute('DELETE FROM macro_rules WHERE series=?',(series,))
        m=len(ds)
        # No pairwise comparisons: O(m * max_period), bounded max_period.
        # Require at least two complete repetitions, and no exceptions.
        candidates=[]
        for p in range(1,min(self.max_period,m//2)+1):
            if any(ds[i]!=ds[i%p] for i in range(p,m)):
                continue
            # Raw m delta symbols; macro p symbols + period + repeat count
            # + anchor = p+3 symbols. Explicit conservative model.
            savings=m-(p+3)
            if savings>=self.min_savings:
                candidates.append((p,savings))
        if not candidates: return
        p,savings=min(candidates,key=lambda z:(-z[1],z[0]))
        self.db.execute('INSERT INTO macro_rules VALUES (?,?,?,?,?)',
                        (series,p,','.join(encode(x) for x in ds[:p]),m,savings))

    def rule(self,series):
        row=self.db.execute('SELECT period,deltas,observations,savings FROM macro_rules WHERE series=?',(series,)).fetchone()
        if row is None:return None
        return {'period':row[0],'deltas':tuple(decode(x) for x in row[1].split(',')),
                'observations':row[2],'symbol_savings':row[3],
                'cost_model':'raw=m delta symbols; macro=p delta symbols + 3 metadata symbols'}

    def predict(self,series):
        rule=self.rule(series)
        if rule is None:return None
        vals=self._values(series)
        d=rule['deltas'][(len(vals)-1)%rule['period']]
        return vals[-1]+d

    def close(self):self.db.close()
