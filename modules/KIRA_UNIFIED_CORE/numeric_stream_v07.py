"""Single-connection deferred numeric learner with batched writes and O(1) last-state lookup.
No geometry, graph edges, floats, or pairwise comparisons. Macros are exact periodic deltas.
"""
import sqlite3
from fractions import Fraction
from contextlib import contextmanager
from numeric_core import rational

def enc(x):
    x=rational(x)
    return f"{x.numerator}/{x.denominator}"

def dec(x):
    a,b=x.split('/')
    return Fraction(int(a),int(b))

class NumericStream:
    def __init__(self,path=':memory:',max_period=16,min_savings=1):
        if max_period<1:raise ValueError('max_period')
        self.db=sqlite3.connect(path)
        self.max_period=max_period;self.min_savings=min_savings
        self.db.executescript("""CREATE TABLE IF NOT EXISTS v07_series(name TEXT PRIMARY KEY,n INTEGER NOT NULL,last TEXT NOT NULL,version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS v07_observations(name TEXT NOT NULL,seq INTEGER NOT NULL,val TEXT NOT NULL,PRIMARY KEY(name,seq));
CREATE TABLE IF NOT EXISTS v07_rules(name TEXT PRIMARY KEY,version INTEGER NOT NULL,period INTEGER NOT NULL,deltas TEXT NOT NULL,savings INTEGER NOT NULL);""")
        self.db.commit()
    def observe(self,name,value):
        value=enc(value)
        row=self.db.execute('SELECT n FROM v07_series WHERE name=?',(name,)).fetchone()
        n=row[0] if row else 0
        self.db.execute('INSERT INTO v07_observations VALUES(?,?,?)',(name,n,value))
        if row:self.db.execute('UPDATE v07_series SET n=?,last=?,version=version+1 WHERE name=?',(n+1,value,name))
        else:self.db.execute('INSERT INTO v07_series VALUES(?,?,?,1)',(name,1,value))
        # version invalidation: no eager rule rebuild, no full-history scan
        self.db.commit()
    def observe_batch(self,name,values):
        with self.db:
            row=self.db.execute('SELECT n FROM v07_series WHERE name=?',(name,)).fetchone()
            n=row[0] if row else 0
            last=None;count=0
            for value in values:
                last=enc(value)
                self.db.execute('INSERT INTO v07_observations VALUES(?,?,?)',(name,n+count,last));count+=1
            if count:
                if row:self.db.execute('UPDATE v07_series SET n=?,last=?,version=version+? WHERE name=?',(n+count,last,count,name))
                else:self.db.execute('INSERT INTO v07_series VALUES(?,?,?,?)',(name,count,last,count))
        return count
    def _rule(self,name):
        row=self.db.execute('SELECT s.version,r.version,r.period,r.deltas,r.savings FROM v07_series s LEFT JOIN v07_rules r ON s.name=r.name WHERE s.name=?',(name,)).fetchone()
        if not row:return None
        if row[1]!=row[0]:return self.consolidate(name)
        if row[2] is None:return None
        return {'period':row[2],'deltas':tuple(map(dec,row[3].split(','))),'symbol_savings':row[4]}
    def consolidate(self,name):
        row=self.db.execute('SELECT version FROM v07_series WHERE name=?',(name,)).fetchone()
        if row is None:return None
        version=row[0]
        vals=[dec(v) for (v,) in self.db.execute('SELECT val FROM v07_observations WHERE name=? ORDER BY seq',(name,))]
        ds=[b-a for a,b in zip(vals,vals[1:])];m=len(ds)
        best=None
        for p in range(1,min(self.max_period,m//2)+1):
            if m-(p+3)<self.min_savings:continue
            if all(ds[i]==ds[i%p] for i in range(p,m)):
                best=(p,ds[:p],m-(p+3));break
        with self.db:
            self.db.execute('DELETE FROM v07_rules WHERE name=?',(name,))
            if best:
                p,pattern,savings=best
                self.db.execute('INSERT INTO v07_rules VALUES(?,?,?,?,?)',(name,version,p,','.join(map(enc,pattern)),savings))
        return None if not best else {'period':best[0],'deltas':tuple(best[1]),'symbol_savings':best[2]}
    def predict(self,name):
        row=self.db.execute('SELECT n,last FROM v07_series WHERE name=?',(name,)).fetchone()
        if not row:return None
        rule=self._rule(name)
        if not rule:return None
        n,last=row
        return dec(last)+rule['deltas'][(n-1)%rule['period']]
    def close(self):self.db.close()
