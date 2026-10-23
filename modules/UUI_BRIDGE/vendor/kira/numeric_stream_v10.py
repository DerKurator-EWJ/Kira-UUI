"""Bounded-working-memory, one-pass periodic macro check; versioned negative cache."""
from numeric_stream_v09 import NumericStreamV09
from numeric_stream_v07 import dec, enc

class NumericStreamV10(NumericStreamV09):
    def consolidate(self,name):
        row=self.db.execute('SELECT version,n FROM v07_series WHERE name=?',(name,)).fetchone()
        if row is None:return None
        version,n=row
        m=n-1
        limit=min(self.max_period,m//2)
        # A candidate needs at least one complete repeat and sufficient symbol savings.
        active=[p for p in range(1,limit+1) if m-(p+3)>=self.min_savings]
        prefixes={p:[] for p in active}
        prev=None; first=[]; i=0
        # Read observations incrementally; do not allocate the full history/delta array.
        for (encoded,) in self.db.execute('SELECT val FROM v07_observations WHERE name=? ORDER BY seq',(name,)):
            val=dec(encoded)
            if prev is not None:
                delta=val-prev
                if i<self.max_period:first.append(delta)
                active=[p for p in active if i<p or delta==first[i%p]]
                i+=1
                if not active:break
            prev=val
        best=None
        if active:
            p=min(active)
            best=(p,first[:p],m-(p+3))
        with self.db:
            self.db.execute('DELETE FROM v07_rules WHERE name=?',(name,))
            if best:
                p,pattern,savings=best
                self.db.execute('INSERT INTO v07_rules VALUES(?,?,?,?,?)',(name,version,p,','.join(map(enc,pattern)),savings))
            else:
                self.db.execute('INSERT INTO v07_rules VALUES(?,?,?,?,?)',(name,version,0,'',0))
        return None if not best else {'period':best[0],'deltas':tuple(best[1]),'symbol_savings':best[2]}
