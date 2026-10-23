# KIRA NUMERIC CORE 0.9 — Workload stress testing

This release adds a **reproducible benchmark and regression suite**, not a new learning architecture. It compares v0.7 deferred, v0.8 deferred, and v0.8 incremental under identical SQLite durability choices and synthetic input sequences. It does **not** claim real-world generalization.

Run: `PYTHONPATH=.:RRM_UNIFIED_0_1 python -m unittest discover -p 'test_*.py' -q` and `PYTHONPATH=.:RRM_UNIFIED_0_1 python benchmark_v09.py`.

Cases include periodic, random irregular, regime-changing, exact rational deltas, 50 independent series, short and long read loads, and single-event versus batch commits. Each case uses separate databases, median of three runs; timing includes write/read operations but not initial database setup, close, or restart. `db_bytes` measures SQLite main database only after close, and does not capture peak RAM or transaction journal bytes. The test checks predictions exist for periodic cases and are absent for irregular and regime changes, but **does not measure predictive accuracy on real data**. `prediction_hits` indicates availability, not quality. Periodicity checks operate on complete sequences, not sliding windows; regime changes correctly invalidate a global periodic rule. Results depend on host, SQLite settings, and filesystem cache. No guarantee of subquadratic end-to-end complexity across all legacy modules.

## Measured issue and targeted fix

The stress suite uncovered a substantial defect in the earlier deferred implementation: when a complete series **has no periodic macro**, every read rescans its entire history because only positive rules were cached. `numeric_stream_v09.py` adds a versioned **negative-result cache** (period=0) to avoid repeated work. Any new observation increments the series version, invalidating both positive and negative caches. This is a small targeted optimization, not a new learning algorithm.
