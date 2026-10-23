# KIRA NUMERIC CORE 0.8 — controlled comparison

`numeric_stream_v08.py` unifies the v0.7 single-connection, batch-capable persistence with either deferred full-history consolidation or bounded incremental period-candidate validation. The same public methods are available for both modes. Mode is chosen explicitly, **not automatically cost-adapted**. Version 0.6 remains available for comparison.

Tests: `PYTHONPATH=.:RRM_UNIFIED_0_1 python -m unittest discover -p 'test_*.py' -q`.
Benchmark: `PYTHONPATH=.:RRM_UNIFIED_0_1 python benchmark_v08.py`.

Important: v0.6 uses a different schema and cost-adaptation logic. Single-event benchmarks have equivalent commit frequency; batch runs exclude v0.6 because it lacks atomic batching. `interleaved` tests reads between write phases. Report **total elapsed time** as primary metric. The model remains a narrow exact periodic-delta learner. Candidate maintenance costs O(P) SQLite queries/updates per new delta, P=max_period. Incremental mode is not guaranteed to outperform deferred mode. SQLite database bytes include metadata and raw observations, not real macro compression. Mode changes of an existing database are not supported without rebuilding candidate state. No auto-switching in v0.8; selecting mode requires a workload-level decision.

## Measured results (local, 5 runs, medians; ms)

| Workload | v0.6 cost-aware | v0.7 deferred | v0.8 deferred | v0.8 incremental |
|---|---:|---:|---:|---:|
| 200 observations, 20 reads, single commits | 25.514 | **12.030** | 12.481 | 16.641 |
| 1000 observations, 200 reads, single commits | 141.389 | 75.467 | **73.302** | 106.617 |
| 1000 observations, 200 reads, interleaved, single commits | 155.029 | **66.268** | 66.403 | 92.452 |
| 1000 observations, 200 reads, atomic batch | n/a | 7.277 | **7.185** | 21.936 |
| 1000 observations, 200 reads, interleaved batches | n/a | **8.036** | 8.240 | 21.867 |

All test data are regular linear sequences. This benchmark is not evidence that the deferred path wins on all possible workloads, and the 0.6 comparison does not isolate algorithmic cost from different DB schema. Batch and single-commit runs have different durability semantics. The extra v0.8 candidates table increases DB size even in deferred mode, so **v0.7 remains preferable as the default** until a use case demonstrates measurable improvement. Full results: `benchmark_v08_results.json`.
