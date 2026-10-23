# KIRA Unified Core 1.2 — Experience & Evidence Layer

Added `evidence_layer.py` and wired optional `fact` into `EvaluatedNumericStream.observe_and_evaluate`.

## Contract
- SQLite persistence on the same database as numeric observations and evaluation history.
- Each piece of evidence has an event ID, source type/ref, confidence, context, subject, predicate, and JSON value.
- Provenance categories: observation, action_result, external_claim, inference, premise.
- Contradictory values for the same subject/predicate and *identical context* create open review cases.
- Neither old nor new evidence is silently deleted or automatically accepted as truth.
- Explicit resolution choices: retain_both, supersede_old, reject_new.
- This is a conservative conflict detector, not a causal reasoner or autonomous evidence adjudicator.
- Evidence submission is optional, preserving the 1.1 API.

## Run tests
From this directory:

    PYTHONPATH=.:RRM_UNIFIED_0_1 python -m unittest discover -p 'test_*.py' -q

Expected: 55 tests (including 4 new evidence tests).

## Example
```python
from evaluation_bridge import EvaluatedNumericStream
k = EvaluatedNumericStream('kira.db')
result = k.observe_and_evaluate('evt-1','sensor',1,success=.5,novelty=.8,
    relevance=.7,information_gain=.5,valence=0,
    fact={'subject':'bed','predicate':'edible','value':False,
          'context':{'room':'A'},'source_type':'action_result',
          'source_ref':'avatar-motor-feedback','confidence':.8})
print(result['evidence'])
print(k.evidence.pending_reviews())
k.close()
```

## Limitations
- Context comparison is exact JSON equality, not semantic context reasoning.
- Value comparison is JSON equality, not semantic equivalence.
- Confidence is stored but does not autonomously determine which evidence wins.
- A single observation can yield multiple reviews when several conflicting hypotheses exist.
- SQLite transactionality across the full numeric-observation/evidence operation is not yet guaranteed; production hardening should use a unified transaction and crash recovery.
- No direct Android/Unity/Godot or autonomous planning integration.
