"""Unified logical RRM arena: stable SQLite handles, bounded LRU wire cache,
shared binary operations, and original evidence engine. No GPU support.
"""
from collections import OrderedDict
from rrm_id import decode, encode
from runtime import BinaryRuntime
from binary_ops import BinaryOps

class UnifiedRuntime(BinaryRuntime):
    def __init__(self, path=':memory:', cache_budget=65536):
        if cache_budget < 0: raise ValueError('cache_budget must be >= 0')
        super().__init__(path)
        self.cache_budget = cache_budget
        self._wires = OrderedDict()
        self._cache_bytes = 0
        self.hits = self.misses = self.bytes_fetched = 0
        self.ops = BinaryOps(self)

    def _remember(self, handle, wire):
        if handle in self._wires:
            old = self._wires.pop(handle)
            self._cache_bytes -= len(old)
        if len(wire) > self.cache_budget:
            return
        while self._cache_bytes + len(wire) > self.cache_budget and self._wires:
            _, old = self._wires.popitem(last=False)
            self._cache_bytes -= len(old)
        self._wires[handle] = wire
        self._cache_bytes += len(wire)

    def _get(self, handle):
        wire = self._wires.get(handle)
        if wire is not None:
            self.hits += 1
            self._wires.move_to_end(handle)
        else:
            self.misses += 1
            row = self.db.execute('SELECT wire FROM records WHERE id=?',(handle,)).fetchone()
            if row is None: raise ValueError('dangling reference')
            wire = row[0]
            self.bytes_fetched += len(wire)
            self._remember(handle, wire)
        rec, end = decode(wire)
        if end != len(wire): raise ValueError('trailing bytes')
        return rec

    def _put(self, rec):
        wire = encode(rec)
        self.db.execute('INSERT OR IGNORE INTO records(wire) VALUES (?)',(wire,))
        handle = self.db.execute('SELECT id FROM records WHERE wire=?',(wire,)).fetchone()[0]
        self._remember(handle, wire)
        return handle

    def cache_stats(self):
        return dict(hits=self.hits, misses=self.misses, bytes_fetched=self.bytes_fetched,
                    resident_records=len(self._wires), resident_wire_bytes=self._cache_bytes,
                    budget_bytes=self.cache_budget)

    def close(self):
        self._wires.clear()
        super().close()
