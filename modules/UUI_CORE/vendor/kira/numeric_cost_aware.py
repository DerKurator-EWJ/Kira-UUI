"""Cost-aware numeric macro learner. No topology, floats, or pairwise scans.

A cheap persistent query count and bounded cost model choose between deferred
and incremental maintenance. The decision is a heuristic, NOT an oracle.
"""
import os
import sqlite3
import tempfile
from fractions import Fraction
from numeric_macros import NumericMacroLearner as Incremental
from numeric_macros_v03 import NumericMacroLearner as Deferred
from numeric_core import rational

class CostAwareNumericLearner:
    def __init__(self, path=':memory:', max_period=16, min_savings=1,
                 predicted_future_reads=100, cost_scan_per_delta=0.01,
                 cost_incremental_per_delta=2.0, cost_replay_per_delta=2.0,
                 min_observations=8):
        if max_period < 1 or predicted_future_reads < 0 or min_observations < 1:
            raise ValueError('invalid cost parameters')
        self._temp=None
        if path==':memory:':
            fd,path=tempfile.mkstemp(prefix='kira_cost_',suffix='.sqlite');os.close(fd);self._temp=path
        self.db=sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS cost_modes (series TEXT PRIMARY KEY, mode TEXT NOT NULL, reads INTEGER NOT NULL DEFAULT 0, writes INTEGER NOT NULL DEFAULT 0, consolidated_at INTEGER NOT NULL DEFAULT -1)")
        self.db.commit()
        self.deferred=Deferred(path,max_period=max_period,min_savings=min_savings)
        # Initializing Incremental on legacy DB would eagerly replay all series.
        # We create it lazily only if a series is promoted.
        self.incremental=None
        self.path=path;self.max_period=max_period;self.min_savings=min_savings
        self.future_reads=predicted_future_reads
        self.scan_cost=cost_scan_per_delta
        self.incremental_cost=cost_incremental_per_delta
        self.replay_cost=cost_replay_per_delta
        self.min_observations=min_observations
    def _fast(self):
        if self.incremental is None:
            self.incremental=Incremental(self.path,self.max_period,self.min_savings)
        return self.incremental
    def _row(self,series):
        row=self.db.execute('SELECT mode,reads,writes,consolidated_at FROM cost_modes WHERE series=?',(series,)).fetchone()
        return row or ('deferred',0,0,-1)
    def mode(self,series):return self._row(series)[0]
    def observe(self,series,value):
        value=rational(value)
        mode,reads,writes,at=self._row(series)
        if mode=='incremental':self._fast().observe(series,value)
        else:self.deferred.observe(series,value)
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO cost_modes VALUES (?,?,?,?,?)',
                            (series,mode,reads,writes+1,at))
    def _n(self,series):
        return self.db.execute('SELECT COUNT(*) FROM macro_observations WHERE series=?',(series,)).fetchone()[0]
    def _should_promote(self,series,n,reads,writes):
        if n < self.min_observations or reads==0:return False
        # Conservative workload forecast: observed reads per write, capped by
        # explicit planning horizon. It never scans historical observations.
        expected=min(self.future_reads, max(0, reads*max(1,n)//max(1,writes)))
        # Deferred: one scan after each future write (if queried).
        # Incremental: one-time replay + incremental work on future writes.
        future_writes=max(1,round(expected*max(1,writes)/max(1,reads)))
        deferred_cost=min(expected,future_writes)*self.scan_cost*n
        promoted_cost=self.replay_cost*n+future_writes*self.incremental_cost*self.max_period
        return deferred_cost>promoted_cost
    def _promote(self,series):
        self._fast()._restore(series)
        with self.db:self.db.execute('UPDATE cost_modes SET mode=? WHERE series=?',('incremental',series))
    def predict(self,series):
        mode,reads,writes,at=self._row(series)
        n=self._n(series)
        if not n:return None
        if mode=='incremental':
            fast=self._fast()
            rule=fast.rule(series)
            if rule is None or rule['observations']!=n-1:fast.consolidate(series)
            result=fast.predict(series)
        else:
            if at!=n:
                self.deferred.consolidate(series)
                at=n
            rule=self.deferred.rule(series)
            if rule is None:result=None
            else:
                # Critical: indexed last record, not _values(series) O(n) scan.
                row=self.db.execute('SELECT val FROM macro_observations WHERE series=? ORDER BY seq DESC LIMIT 1',(series,)).fetchone()
                last=rational(row[0]); result=last+rule['deltas'][(n-1)%rule['period']]
        reads+=1
        with self.db:self.db.execute('INSERT OR REPLACE INTO cost_modes VALUES (?,?,?,?,?)',(series,mode,reads,writes,at))
        # Promotion only AFTER producing the result, never on a cold first read.
        if mode=='deferred' and self._should_promote(series,n,reads,writes):self._promote(series)
        return result
    def consolidate(self,series):
        if self.mode(series)=='incremental':return self._fast().consolidate(series)
        result=self.deferred.consolidate(series)
        with self.db:self.db.execute('UPDATE cost_modes SET consolidated_at=? WHERE series=?',(self._n(series),series))
        return result
    def close(self):
        if self.incremental:self.incremental.close()
        self.deferred.close();self.db.close()
        if self._temp:os.unlink(self._temp)
