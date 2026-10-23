"""Unified SQLite stream: deferred or bounded incremental period validation.
Exact rational arithmetic; no graph/topology; no historical scans on hot reads.
The caller selects durability (single-event commit or atomic batch).
"""
from numeric_stream_v07 import NumericStream,enc,dec

class NumericStreamV08(NumericStream):
    def __init__(self,path=':memory:',max_period=16,min_savings=1,mode='deferred'):
        if mode not in ('deferred','incremental'):raise ValueError('mode')
        super().__init__(path,max_period,min_savings)
        self.mode=mode
        self.db.execute("""CREATE TABLE IF NOT EXISTS v08_candidates(
            name TEXT NOT NULL,p INTEGER NOT NULL,pattern TEXT NOT NULL,
            valid INTEGER NOT NULL,PRIMARY KEY(name,p))""")
        self.db.commit()
    def _advance(self,name,value,n,last):
        if n==0:return
        delta=enc(dec(value)-dec(last))
        m=n-1 # number of deltas before this event
        for p in range(1,self.max_period+1):
            row=self.db.execute('SELECT pattern,valid FROM v08_candidates WHERE name=? AND p=?',(name,p)).fetchone()
            if row is None:
                self.db.execute('INSERT INTO v08_candidates VALUES (?,?,?,1)',(name,p,delta))
            elif row[1]:
                parts=row[0].split(',')
                if m<p:parts.append(delta)
                elif parts[m%p]!=delta:
                    self.db.execute('UPDATE v08_candidates SET valid=0 WHERE name=? AND p=?',(name,p))
                    continue
                self.db.execute('UPDATE v08_candidates SET pattern=? WHERE name=? AND p=?',(','.join(parts),name,p))
    def observe_batch(self,name,values):
        with self.db:
            row=self.db.execute('SELECT n,last FROM v07_series WHERE name=?',(name,)).fetchone()
            n,last=row if row else (0,None)
            count=0
            for value in values:
                v=enc(value)
                if self.mode=='incremental':self._advance(name,v,n+count,last)
                self.db.execute('INSERT INTO v07_observations VALUES(?,?,?)',(name,n+count,v))
                last=v;count+=1
            if count:
                if row:self.db.execute('UPDATE v07_series SET n=?,last=?,version=version+? WHERE name=?',(n+count,last,count,name))
                else:self.db.execute('INSERT INTO v07_series VALUES(?,?,?,?)',(name,count,last,count))
        return count
    def observe(self,name,value):return self.observe_batch(name,(value,))
    def consolidate(self,name):
        if self.mode=='deferred':return super().consolidate(name)
        row=self.db.execute('SELECT n,version FROM v07_series WHERE name=?',(name,)).fetchone()
        if row is None:return None
        n,version=row;m=n-1
        best=None
        for p,pattern in self.db.execute('SELECT p,pattern FROM v08_candidates WHERE name=? AND valid=1 AND p<=? ORDER BY p',(name,m//2)):
            savings=m-(p+3)
            if savings>=self.min_savings:
                best=(p,pattern,savings);break
        with self.db:
            self.db.execute('DELETE FROM v07_rules WHERE name=?',(name,))
            if best:self.db.execute('INSERT INTO v07_rules VALUES(?,?,?,?,?)',(name,version,*best))
        if best:return {'period':best[0],'deltas':tuple(map(dec,best[1].split(','))),'symbol_savings':best[2]}
        return None
    def predict(self,name):
        # Use inherited cached version checking; no history read in incremental mode.
        return super().predict(name)
