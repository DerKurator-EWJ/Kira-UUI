# KIRA NUMERIC CORE 0.4 — incremental macro experiment

Builds on v0.3. Replaces `numeric_macros.py` with an incremental candidate tracker. Previous implementation retained as `numeric_macros_v03.py` for reproducible comparisons.

## Changes
- Persistent `macro_state` tracks latest rational value and delta count; prediction no longer reads full history.
- Persistent `macro_candidates` maintains up to `max_period` exact periodic candidate prefixes; each new delta checks at most that many candidates. An invalidated candidate remains invalid for the complete series. No graph edges, nodes or spatial topology.
- Existing v0.3 databases can be upgraded by a one-time replay of historical observations.
- New observations invalidate previously consolidated rules until `consolidate` is called again.
- `max_period` is fixed per database/series; changing it on reopen is not supported without rebuilding candidate state.

## Measured locally (3 trials, median, 8 candidate periods, same SQLite default transaction behavior)

| Deltas | Version | Append total ms | Consolidate ms | Prediction ms/call | SQLite bytes |
|---:|---|---:|---:|---:|---:|
| 200 | v0.3 | 11.30 | 0.76 | 0.2094 | 28672 |
| 200 | v0.4 | 20.58 | 0.22 | 0.0121 | 45056 |
| 1000 | v0.3 | 59.52 | 3.39 | 1.0213 | 65536 |
| 1000 | v0.4 | 105.65 | 0.22 | 0.0129 | 81920 |
| 3000 | v0.3 | 165.14 | 9.27 | 3.2651 | 147456 |
| 3000 | v0.4 | 296.71 | 0.27 | 0.0113 | 163840 |

**Tradeoff:** v0.4 is substantially faster for prediction and consolidation, but slower for ingestion and uses more storage. It is NOT a universal performance improvement. The prototype stores original observations and candidate metadata, so compression is symbolic, not physical. `O(P)` SQL statements per appended delta, with bounded P; DB transaction overhead remains substantial. `O(1)` refers only to latest-state lookup, not full wall-clock work or arbitrary-precision arithmetic. No hard worst-case time guarantee from SQLite.

## Run
From package directory:
```
PYTHONPATH=.:RRM_UNIFIED_0_1 python -m unittest discover -p 'test*.py' -q
python benchmark_macros.py
```
22 tests passed locally. An existing ResourceWarning in `numeric_core.py` may appear in some runs.

## Next optimization
Use batch inserts / in-memory candidate cache with transactional checkpoints and validate restart recovery. Then test workload-dependent dispatch between incremental and deferred consolidation rather than always paying the v0.4 append overhead.
