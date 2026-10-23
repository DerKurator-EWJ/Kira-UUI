from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys
import tempfile

from equinox_forge.unified_numeric_kernel import (
    UNIFIED_NUMERIC_KERNEL_VERSION,
    load_unified_numeric_kernel_pack,
    run_unified_numeric_kernel_demo,
)

assert UNIFIED_NUMERIC_KERNEL_VERSION == "0.55"

with tempfile.TemporaryDirectory() as td:
    out = Path(td) / "trial"
    report = run_unified_numeric_kernel_demo(out)
    assert all(report["controls"].values()), report["controls"]
    assert report["parity"]["failure_count"] == 0
    assert report["parity"]["query_pairs_compared"] == 240
    assert report["parity"]["resolved_query_pairs"] == 97
    assert report["route_count"] == 15
    assert report["concept_shortcut_count"] == 24
    assert report["pack_sizes"]["fraction_of_nested"] < 1.0

    pack_path = out / "kernel.eqnum.json"
    pack = load_unified_numeric_kernel_pack(pack_path)
    assert pack.digest == report["pack_digest"]

    # A fresh runtime-only process must not load any v0.48-v0.54 runtime layer.
    code = r'''
import json, sys
from equinox_forge.unified_numeric_kernel import load_unified_numeric_kernel_pack, UnifiedNumericExecutionKernel
p = load_unified_numeric_kernel_pack(sys.argv[1])
k = UnifiedNumericExecutionKernel(p)
old = [m for m in sys.modules if m in {
    "equinox_forge.numeric_runtime",
    "equinox_forge.macro_numeric_runtime",
    "equinox_forge.canonical_relation_facets",
    "equinox_forge.relation_macro_runtime",
    "equinox_forge.unified_relation_runtime",
}]
r = p.routes[0]
subjects = sorted({f.subject_id for f in p.facts})
res = k.resolve(subjects[0], r.query_relation_id)
print(json.dumps({"historical": old, "status": res.status, "routes": len(p.routes)}))
'''
    proc = subprocess.run(
        [sys.executable, "-c", code, str(pack_path)],
        check=True, capture_output=True, text=True,
    )
    isolated = json.loads(proc.stdout)
    assert isolated["historical"] == []
    assert isolated["routes"] == 15

    payload = json.loads(pack_path.read_text(encoding="utf-8"))
    payload["facts"][0][2] += 1
    bad = out / "tampered.eqnum.json"
    bad.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    try:
        load_unified_numeric_kernel_pack(bad)
    except ValueError as exc:
        assert "digest mismatch" in str(exc)
    else:
        raise AssertionError("tampered v0.55 kernel pack was accepted")

print("FORGE_V055_UNIFIED_NUMERIC_KERNEL_OK")
