from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).name


def discover(pattern: str = "test_*.py") -> list[Path]:
    return sorted(p for p in (ROOT / "tests").glob(pattern) if p.name != THIS_FILE)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run every Equinox Forge regression test in an isolated process.")
    ap.add_argument("--pattern", default="test_*.py")
    ap.add_argument("--timeout", type=float, default=180.0, help="per-test timeout in seconds")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    tests = discover(args.pattern)
    env = os.environ.copy()
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + old if old else "")

    rows = []
    started = time.perf_counter()
    for path in tests:
        t0 = time.perf_counter()
        status = "PASS"
        returncode = 0
        error = None
        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                cwd=ROOT,
                env=env,
                timeout=args.timeout,
                check=False,
            )
            returncode = proc.returncode
            if returncode != 0:
                status = "FAIL"
        except subprocess.TimeoutExpired:
            status = "TIMEOUT"
            returncode = 124
            error = f"exceeded {args.timeout:.1f}s"
        elapsed = time.perf_counter() - t0
        row = {
            "test": path.name,
            "status": status,
            "returncode": returncode,
            "elapsed_s": round(elapsed, 6),
        }
        if error:
            row["error"] = error
        rows.append(row)
        print(f"[{status:7}] {path.name:<45} {elapsed:8.3f}s", flush=True)

    total = time.perf_counter() - started
    passed = sum(r["status"] == "PASS" for r in rows)
    failed = len(rows) - passed
    summary = {
        "format": "equinox-forge-regression-summary-v1",
        "version": "0.30",
        "tests": len(rows),
        "passed": passed,
        "failed": failed,
        "elapsed_s": round(total, 6),
        "results": rows,
    }
    print(f"\nEQUINOX_FORGE_REGRESSION tests={len(rows)} passed={passed} failed={failed} elapsed={total:.3f}s")

    if args.json_out:
        out = Path(args.json_out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(out)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
