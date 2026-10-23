# KIRA Unified Numeric Core + Decision Matrix 1.1 (experimental)

Contains the **unchanged original** KIRA Numeric Core 1.0 Python files and the **unchanged original** Decision Matrix 0.1 Kotlin source/tests under `decision_matrix_kotlin/`. Two added Python files connect the numeric stream to a Python port of the matrix: `decision_matrix.py` and `evaluation_bridge.py`.

## Use
```python
from evaluation_bridge import EvaluatedNumericStream
core = EvaluatedNumericStream('kira.db')
result = core.observe_and_evaluate('event-1', 'sensor-series', 12,
    success=0.8, novelty=0.5, relevance=0.7,
    information_gain=0.4, valence=0.6)
print(result['utterance'])
core.close()
```
Run: `PYTHONPATH=.:RRM_UNIFIED_0_1 python -m unittest discover -p 'test_*.py' -q`.

## Boundaries
- Numeric Core 1.0 is **Python**, Matrix 0.1 is **Kotlin**. This package adds a **Python port**, not a JVM-to-Python runtime bridge or Android integration.
- The adapter obtains **surprise** from numeric predictions only when available. No prediction => surprise 0 and confidence 0, not a claim that the result was expected.
- Caller must supply success, novelty, relevance, information gain and valence; these are **not learned** by this adapter. Thresholds are heuristics, not calibrated.
- Evaluation journal is persisted in SQLite; evaluation + observation are **not atomic across both writes** in this baseline. Crash recovery and migration are future work.
- Prediction patterns are exact periodic deltas. These are **not** a general world model, causal reasoning, intrinsic curiosity, emotions, or autonomous decision-making.
- No object-specific labels, no numeric-ID proximity, no forced memory revision on contradiction.
