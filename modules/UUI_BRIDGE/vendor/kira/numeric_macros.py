"""Incremental numeric periodic macros, exact rationals, no graph primitives.
Append O(1) SQLite statements per event, bounded O(P) pattern checks.
Period candidates are verified from the first delta and invalidated permanently
on mismatch; periodic prefixes are not reaccepted after an exception.
"""
import sqlite3
from fractions import Fraction
from numeric_core import rational

def encode(v):
    v=rational(v)
    return f"{v.numerator}/{v.denominator}"
def decode(s):
    a,b=s.split('/')
    return Fraction(int(a),int(b))

class NumericMacroLearner:
    def __init__(self,path=':memory:',max_period=16,min_savings=1):
        if max_period<1: raise ValueError('max_period must be positive')
        self.max_period=max_period; self.min_savings=min_savings
        self.db=sqlite3.connect(path)
        self.db.executescript('''CREATE TABLE IF NOT EXISTS macro_observations(series TEXT NOT NULL,seq INTEGER NOT NULL,val TEXT NOT NULL,PRIMARY KEY(series,seq));
CREATE TABLE IF NOT EXISTS macro_rules(series TEXT PRIMARY KEY,period INTEGER NOT NULL,deltas TEXT NOT NULL,observations INTEGER NOT NULL,savings INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS macro_state(series TEXT PRIMARY KEY,n INTEGER NOT NULL,last TEXT NOT NULL,delta_count INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS macro_candidates(series TEXT NOT NULL,p INTEGER NOT NULL,pattern TEXT NOT NULL,valid INTEGER NOT NULL,PRIMARY KEY(series,p));
''')
        self.db.commit()
        # Upgrade legacy databases by replaying stored observations once per series.
        for (series,) in self.db.execute('SELECT DISTINCT series FROM macro_observations WHERE series NOT IN (SELECT series FROM macro_state)').fetchall():
            self._restore(series)
    def _restore(self,series):
        vals=[decode(x) for (x,) in self.db.execute('SELECT val FROM macro_observations WHERE series=? ORDER BY seq',(series,))]
        if not vals:return
        with self.db:
            self.db.execute('DELETE FROM macro_candidates WHERE series=?',(series,))
            self.db.execute('INSERT OR REPLACE INTO macro_state VALUES (?,?,?,?)',(series,1,encode(vals[0]),0))
            for v in vals[1:]: self._advance(series,v,store=False)
            self._update_rule(series)
    def _advance(self,series,v,store=True):
        n,last,m=self.db.execute('SELECT n,last,delta_count FROM macro_state WHERE series=?',(series,)).fetchone()
        d=encode(v-decode(last))
        if store: self.db.execute('INSERT INTO macro_observations VALUES (?,?,?)',(series,n,encode(v)))
        # Candidate p stores its first p deltas; then checks only delta at m%p.
        for p in range(1,self.max_period+1):
            row=self.db.execute('SELECT pattern,valid FROM macro_candidates WHERE series=? AND p=?',(series,p)).fetchone()
            if row is None:
                pattern=[d]; valid=1
                self.db.execute('INSERT INTO macro_candidates VALUES (?,?,?,1)',(series,p,d))
            else:
                pattern=row[0].split(',');valid=row[1]
                if not valid: continue
                if m<p: pattern.append(d)
                elif pattern[m%p]!=d: valid=0
                self.db.execute('UPDATE macro_candidates SET pattern=?,valid=? WHERE series=? AND p=?',(','.join(pattern),valid,series,p))
        self.db.execute('UPDATE macro_state SET n=?,last=?,delta_count=? WHERE series=?',(n+1,encode(v),m+1,series))
    def observe(self,series,value):
        v=rational(value)
        with self.db:
            row=self.db.execute('SELECT n FROM macro_state WHERE series=?',(series,)).fetchone()
            if row is None:
                self.db.execute('INSERT INTO macro_observations VALUES (?,?,?)',(series,0,encode(v)))
                self.db.execute('INSERT INTO macro_state VALUES (?,?,?,0)',(series,1,encode(v)))
            else:
                self._advance(series,v)
                self.db.execute('DELETE FROM macro_rules WHERE series=?',(series,))
        return None
    def _update_rule(self,series):
        row=self.db.execute('SELECT delta_count FROM macro_state WHERE series=?',(series,)).fetchone()
        if row is None:return
        m=row[0]; options=[]
        for p,pattern,valid in self.db.execute('SELECT p,pattern,valid FROM macro_candidates WHERE series=? AND valid=1 AND p<=?',(series,m//2)):
            savings=m-(p+3)
            if savings>=self.min_savings:options.append((savings,p,pattern))
        if not options:
            self.db.execute('DELETE FROM macro_rules WHERE series=?',(series,));return
        savings,p,pattern=max(options,key=lambda x:(x[0],-x[1]))
        self.db.execute('INSERT OR REPLACE INTO macro_rules VALUES (?,?,?,?,?)',(series,p,pattern,m,savings))
    def consolidate(self,series):
        with self.db:self._update_rule(series)
        return self.rule(series)
    def rule(self,series):
        row=self.db.execute('SELECT period,deltas,observations,savings FROM macro_rules WHERE series=?',(series,)).fetchone()
        if row is None:return None
        return {'period':row[0],'deltas':tuple(map(decode,row[1].split(','))),'observations':row[2],
                'symbol_savings':row[3], 'cost_model':'raw=m delta symbols; macro=p delta symbols + 3 metadata symbols'}
    def predict(self,series):
        rule=self.rule(series)
        if rule is None:return None
        row=self.db.execute('SELECT last,delta_count FROM macro_state WHERE series=?',(series,)).fetchone()
        if row is None:return None
        return decode(row[0])+rule['deltas'][row[1]%rule['period']]
    def close(self):self.db.close()
