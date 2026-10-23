"""Deferred numeric learner with persisted negative-rule cache.
Caches the fact that a series has no valid periodic macro for its current version.
"""
from numeric_stream_v07 import NumericStream,dec

class NumericStreamV09(NumericStream):
    def _rule(self,name):
        row=self.db.execute('SELECT s.version,r.version,r.period,r.deltas,r.savings FROM v07_series s LEFT JOIN v07_rules r ON s.name=r.name WHERE s.name=?',(name,)).fetchone()
        if row is None:return None
        if row[1]!=row[0]:
            result=self.consolidate(name)
            if result is None:
                with self.db:
                    self.db.execute('INSERT OR REPLACE INTO v07_rules VALUES(?,?,?,?,?)',(name,row[0],0,'',0))
            return result
        if not row[2]:return None
        return {'period':row[2],'deltas':tuple(map(dec,row[3].split(','))),'symbol_savings':row[4]}
