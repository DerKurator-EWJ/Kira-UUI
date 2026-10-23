# UUI Core 0.3 — Willi-Syntax + UUIX-IR

UUI 0.3 supports: `was_brauchst_du` (if), `geht_das` (else), `hier` (print), `es_sei` (let), `so_nicht` (set), `mach` (fn), `her_damit` (return), `solange` (while), `pass_uff` (observe). The existing `resolve` and `as` syntax remains. Old English spellings are retained as compatibility aliases. Mathematical operators are unchanged.

## Windows / Python 3.14

```powershell
py -3.14 run_uui.py willi_demo.uui
py -3.14 run_uui.py willi_demo.uui --compile
py -3.14 run_uui.py willi_demo.uuix
py -3.14 -m unittest discover -v
```

## Honest technical status

`.uuix` is currently a **versioned, integrity-checked JSON AST container**, not compact binary bytecode or a standalone virtual machine. It runs through the existing Python interpreter (`run_compiled`), so Python is still required. This is a first compiler pipeline, not yet platform-neutral native execution. The SHA-256 checksum detects accidental corruption; it is **not a signature or security boundary**. Do not run untrusted UUIX packages. Runtime step/depth budgets exist but are not a complete sandbox. Forge's real-pack end-to-end behavior and Python 3.14 compatibility are not yet independently tested.
