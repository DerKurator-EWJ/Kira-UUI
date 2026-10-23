# UUI Core 0.5 — Evidence & Inquiry

Builds on UUI Core 0.4, preserving the Willi syntax, UUIX JSON IR, Kira 1.4 and Forge 0.56 adapters.

## New implemented library operations

- `lern_draus(subject, predicate, value, source_ref, event_id)` — **records an external claim**, not verified learning. Duplicate event IDs rejected; contradictory claims open a review without deleting either record.
- `was_weisst_du(subject, predicate)` — returns JSON text with stored evidence and belief statuses. `known` means records exist, not that they are true.
- `woher_weisst_du(subject, predicate)` — returns JSON text with provenance, confidence and statuses.
- `warum_nicht(subject, predicate)` — creates/deduplicates a persistent unknown-relation inquiry. The original one-argument template remains compatible.
- `was_waere_wenn(initial, operand, operation)` — pure rational arithmetic (`plus`, `minus`, `mal`, `geteilt`), **not** a world simulator.

These are bounded adapters, not autonomous reasoning, validated knowledge, sensor perception or model training. Returned JSON is text because UUI lacks native records. Runtime is Python, UUIX is still JSON IR, not native bytecode.

## Run

```powershell
py -3.14 run_uui.py evidence_demo.uui
py -3.14 -m unittest discover -v
```

Tested in this environment with Python 3.11+/3.13? Check the environment version from `python --version`; Python 3.14.7 remains untested here.

Default Kira DB is in-memory, so evidence persists for the bridge's lifetime only. For persistence pass a file path to `UUIBridge(kira_db='kira.sqlite')` in a Python host.

## Caution

`lern_draus` currently accepts strings as claims, with `source_type='external_claim'` and confidence defaulting to 1.0 as provided by the underlying evidence layer. This is **not a calibrated confidence** and should not be interpreted as truth. A future release should require explicit source type, context and confidence and expose a typed evidence result.

## Version 0.7

Siehe `README_07.md`. Drei neue Funktionen zur **kontrollierten** Evidenzrevision mit Audit-Trail; 52 Tests bestanden (Python-Version der Laufzeitumgebung). Keine automatische Wahrheitsentscheidung oder selbstständiges Modelltraining.
