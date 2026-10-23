# KIRA Unified Core 1.4 — Experiment Planner

## Scope
`experiment_planner.py` adds a persistent SQLite-backed experiment proposal layer to the existing InquiryController and EvidenceLayer. It ranks **registered** capabilities by inquiry priority, estimated information value, declared risk and cost. Proposals are idempotent while pending, require explicit approval, and **do not execute** physical or virtual actions. A caller can submit a verified result through `complete()`, which stores provenance-aware evidence and generates contradiction inquiries.

## Example
```python
import sqlite3
from evidence_layer import EvidenceLayer
from inquiry_controller import InquiryController
from experiment_planner import ExperimentPlanner

conn = sqlite3.connect('kira.db')
evidence = EvidenceLayer(conn)
inquiry = InquiryController(evidence)
planner = ExperimentPlanner(inquiry)
planner.register_capability('inspect_edibility', 'edible', 'Inspect material in simulation', risk=0.1)
question_id = inquiry.ask_about('bed', 'edible')
proposal = planner.propose(question_id)[0]
planner.approve(proposal['experiment_id'])
# External simulator runs the approved action and returns a VERIFIED observation.
planner.complete(proposal['experiment_id'], event_id='sim-001', value=False, source_ref='simulator-run-001')
```

## Limits
- Information gain is a documented heuristic, not a learned probabilistic model.
- Capabilities must be registered; no novel motor plans or experiments are invented.
- No execution adapter or autonomous actuation is supplied.
- SQLite persistence is local. Cross-component transactions and concurrent writers need hardening.
- This package is Python; no Android/Godot integration is claimed.

## Tests
From the package directory:
`PYTHONPATH=.:RRM_UNIFIED_0_1 python -m unittest discover -p 'test_*.py' -q`
