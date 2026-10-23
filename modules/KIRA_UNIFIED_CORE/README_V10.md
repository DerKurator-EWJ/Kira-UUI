# KIRA NUMERIC CORE 1.0 — streaming consolidation & scaling

This is an experimental numeric submodule, **not** a Kira-wide v1.0 release. `numeric_stream_v10.py` replaces full-history materialization during macro consolidation with a single SQLite cursor and a bounded number of active period candidates (max_period). It retains v0.9 versioned negative-result caching. No graph, spatial interpretation, or opaque-ID similarity is assumed.

Run: `PYTHONPATH=.:RRM_UNIFIED_0_1 python -m unittest discover -p 'test_*.py' -q` and `PYTHONPATH=.:RRM_UNIFIED_0_1 python benchmark_v10.py`.

Complexity: O(n * P) worst-case candidate checks for fixed maximum period P, O(P) extra Python working memory for consolidation (ignoring SQLite internal buffers), and O(n) persistent observation storage. No claim of O(1) writes or universally subquadratic behavior across legacy modules. First query after data changes can still cost O(n); repeat queries are cached. The test measures Python allocations during consolidation only; it is not total process RSS. Full observation retention means no database compression. Synthetic patterns do not validate general intelligence or forecasting accuracy.

## Observed measurements

47/47 unit tests passed. In single-run local synthetic benchmarks, periodic 1M observations: v0.9 first read 8.78s and ~152.9 MB peak Python allocations, v1.0 streaming first read 12.08s and ~4.8 KB peak Python allocations. Thus streaming **reduces consolidation allocation dramatically but is slower** for this regular case. Irregular 100k: v0.9 first read 0.659s, v1.0 0.0007s (early mismatch exit); v1.0 irregular 1M 0.0007s. The irregular 1M v0.9 run exceeded timeout with tracemalloc and is omitted. These are single measurements, not reliable performance medians. Main database sizes at 1M were ~38.8 MB periodic and ~34.4 MB irregular, with no compression.
