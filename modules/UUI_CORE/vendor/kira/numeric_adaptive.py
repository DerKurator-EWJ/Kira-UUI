"""Adaptive write-light vs read-optimized numeric macro learner.
Read-heavy series can be promoted; cold series use the v0.3 deferred learner.
Promotion costs a one-time O(n*P) replay; all accesses are exact rational.
"""
import sqlite3
from numeric_macros import NumericMacroLearner
from numeric_macros_v03 import NumericMacroLearner as DeferredLearner

class AdaptiveNumericLearner:
    def __init__(self, path=':memory:', max_period=16, promote_reads=8, min_savings=1):
        if promote_reads < 1: raise ValueError('promote_reads must be positive')
        self.db=sqlite3.connect(path)
        self.db.execute('CREATE TABLE IF NOT EXISTS adaptive_modes(series TEXT PRIMARY KEY, mode TEXT NOT NULL, reads INTEGER NOT NULL DEFAULT 0)')
        self.db.commit()
        self.max_period=max_period
        self.promote_reads=promote_reads
        self.min_savings=min_savings
        # Shared database file is used only for file-backed databases.
        # In-memory mode uses a private temporary SQLite file so both learners share state.
        import tempfile, os
        self._tmp=None
        if path==':memory:':
            fd,self._tmp=tempfile.mkstemp(prefix='kira_adaptive_',suffix='.db');os.close(fd)
            self.db.close(); self.db=sqlite3.connect(self._tmp)
            self.db.execute('CREATE TABLE IF NOT EXISTS adaptive_modes(series TEXT PRIMARY KEY, mode TEXT NOT NULL, reads INTEGER NOT NULL DEFAULT 0)')
            self.db.commit()
            path=self._tmp
        self.fast=NumericMacroLearner(path,max_period=max_period,min_savings=min_savings)
        self.slow=DeferredLearner(path,max_period=max_period,min_savings=min_savings)
    def _state(self,series):
        row=self.db.execute('SELECT mode,reads FROM adaptive_modes WHERE series=?',(series,)).fetchone()
        return row or ('deferred',0)
    def mode(self,series):return self._state(series)[0]
    def observe(self,series,value):
        mode,_=self._state(series)
        if mode=='incremental': return self.fast.observe(series,value)
        return self.slow.observe(series,value)
    def _promote(self,series):
        # The v0.4 replay reconstructs incremental state from existing observations.
        # Since its constructor only replays absent macro_state, trigger explicitly.
        self.fast._restore(series)
        with self.db:self.db.execute('INSERT OR REPLACE INTO adaptive_modes VALUES (?,?,0)',(series,'incremental'))
    def _ensure_rule(self,series,learner):
        row=self.db.execute('SELECT COUNT(*) FROM macro_observations WHERE series=?',(series,)).fetchone()
        n=row[0]
        rule=learner.rule(series)
        if n and (rule is None or rule['observations']!=n-1):
            learner.consolidate(series)

    def predict(self,series):
        mode,reads=self._state(series)
        if mode=='incremental':
            self._ensure_rule(series,self.fast)
            return self.fast.predict(series)
        reads+=1
        if reads>=self.promote_reads:
            self._promote(series)
            self._ensure_rule(series,self.fast)
            return self.fast.predict(series)
        with self.db:self.db.execute('INSERT OR REPLACE INTO adaptive_modes VALUES (?,?,?)',(series,'deferred',reads))
        # Deferred prediction uses the original v0.3 implementation.
        self._ensure_rule(series,self.slow)
        return self.slow.predict(series)
    def consolidate(self,series):
        if self.mode(series)=='incremental':return self.fast.consolidate(series)
        return self.slow.consolidate(series)
    def close(self):
        self.slow.close();self.fast.close();self.db.close()
        if self._tmp:
            import os
            os.unlink(self._tmp)
