# KIRA NUMERIC CORE 0.3 — cost-gated numeric macros (experimental)

Builds on 0.2. New `numeric_macros.py` offers an **optional** persistent macro candidate for exact periodic rational *differences*. No spatial topology, graph nodes, edges or geometry are assumed. Existing learners are unchanged.

## Contract
- `observe(series, value)` persists a rational observation; **does not rescan the entire series**.
- `consolidate(series)` tests periods 1..`max_period` with a single scan per period, bounded `O(m * max_period)` time. Use periodically or offline, **not after each observation**.
- A macro is accepted only if its delta pattern repeats without exceptions for at least two full periods and saves at least `min_savings` symbols in the explicitly stated model: raw `m` deltas vs `p` delta symbols + 3 metadata symbols. This is **not** an empirical byte/CPU win.
- `predict(series)` extrapolates the periodic pattern; it is a hypothesis, not established truth.
- SQLite keeps raw observations and macro metadata. Thus this version **does not reduce on-disk size**. Compression / replacement of raw history and incremental candidate maintenance remain future work.
- `predict` currently reads series history to obtain the last value; for long histories it is **O(m)**. A persistent last-value table would remove that overhead.
- `consolidate` on every new observation would again yield quadratic cumulative cost: explicitly avoid that.

## Tests
Run from this directory: `PYTHONPATH=.:RRM_UNIFIED_0_1 python -m unittest -q test_numeric_macros test_numeric_patterns test_numeric_core test_integration`

This is a verified **candidate discovery and correctness** experiment, not a claim of production-ready efficiency or generalized learning.
