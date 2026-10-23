# UUI Core 0.2 — Experimental interpreter

UUI 0.2 adds `fn`/`return`, `if`/`else`, bounded `while`, `set`, booleans, comparisons and boolean operators to UUI 0.1. Existing `let`, `print`, `observe`, `resolve` and exact rational arithmetic remain.

## Requirements

Python 3.10+ (designed to use the standard library only; tested in the build environment with Python 3.13.5, **not** directly tested with Python 3.14.7). No pip dependencies required for the included demo.

## Windows / PowerShell

Unzip the archive and open PowerShell in the `UUI_CORE_0_2` folder:

```powershell
py -3.14 run_uui.py control_flow.uui
py -3.14 -m unittest discover -v
```

If the Python launcher isn't installed, use `python` instead of `py -3.14`.

Expected demo output:

```
UUI 0.2 OK
{"id": "kira.numeric:1", "rational": ["30", "1"]}
```

Use `--database path/to/kira.sqlite` to persist numeric observations (the default is in-memory). `--forge-pack` optionally supplies an existing Forge kernel pack; full Forge pack compatibility is not validated by this demo.

## Scope and limitations

- This is a custom *interpreted syntax* implemented in Python, **not** native or platform-neutral UUI bytecode (`.uuix`).
- `while` loops, expressions and recursion have configurable runtime budgets. These are basic runaway-execution safeguards, **not** a hardened sandbox against hostile code or denial of service.
- Functions use local parameters and local variables, no closures or global access. A function without `return` yields an internal `None` value; operations on it are unsupported.
- `if` and `while` blocks share the current variable environment. `let` declares, `set` changes existing values.
- Host imports, arbitrary filesystem operations, and dynamic Python evaluation are not exposed through UUI syntax.
- Bundled Kira/Forge code is retained as Python vendor modules; no automatic port to a native runtime has occurred.

## Files

- `uui_lang.py` — lexer, parser, interpreter
- `run_uui.py` — CLI
- `hello.uui` — 0.1 compatibility demo
- `control_flow.uui` — 0.2 demo
- `test_uui.py`, `test_uui_02.py`, `test_bridge.py` — tests
- `uui_bridge.py`, `vendor/` — existing integration components
