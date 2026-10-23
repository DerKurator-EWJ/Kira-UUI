# KIRA NUMERIC CORE 0.7 — Single-connection / Batch Experiment

Adds `numeric_stream_v07.py`: one SQLite connection, persisted last-value and length per series, version-based rule invalidation, and `observe_batch()` for one transaction per batch. Exact rational values; no graph or geometric primitives. The existing v0.6 cost-aware runtime is **preserved** and is not replaced automatically. This is a specialized deferred execution path.

`python -m unittest discover -p "test_*.py"` and `python benchmark_v07.py`.

Caveats: `predict()` after a write still triggers full-history consolidation O(n·P) (P fixed maximum period); after consolidation it uses indexed metadata and cached rule. Every `observe()` commits separately for durability. `observe_batch()` is faster but trades event-by-event durability for atomic batch durability. Macro rules are strict periodic patterns, not general learned laws. Compare equal tasks and report both training and prediction costs.

Local benchmark (5 runs, median; 1,000 values, 200 predictions, Python/SQLite; milliseconds):

| Path | Write | Read | Sum of median phases | DB |
|---|---:|---:|---:|---:|
| v0.3 deferred | 59.113 | 232.002 | 291.115 | 53,248 B |
| v0.7 individual commits | 60.875 | 4.505 | 65.380 | 61,440 B |
| v0.7 atomic batch | 2.519 | 4.138 | 6.657 | 61,440 B |

These are **not** comparisons against v0.6 cost-aware strategy; benchmark contrasts equivalent periodic sequence operations with v0.3 deferred. Batch semantics differ from per-event durability. This is a local microbenchmark, not a production or whole-Kira result. Sum of medians differs from median of per-run totals.

Run full tests with `PYTHONPATH=RRM_UNIFIED_0_1 python -m unittest discover -p 'test_*.py' -q`.
