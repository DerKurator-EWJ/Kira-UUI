from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys
import tempfile

import equinox_forge
from equinox_forge.direct_teacher_flat_compiler import (
    DIRECT_TEACHER_FLAT_COMPILER_VERSION,
    run_direct_teacher_flat_compiler_trial,
)
from equinox_forge.unified_numeric_kernel import (
    UNIFIED_NUMERIC_KERNEL_VERSION,
    load_unified_numeric_kernel_pack,
)

assert equinox_forge.__version__ == "0.56.0"
assert DIRECT_TEACHER_FLAT_COMPILER_VERSION == "0.56"
assert UNIFIED_NUMERIC_KERNEL_VERSION == "0.55"  # compiler change, runtime format unchanged

with tempfile.TemporaryDirectory() as td:
    out = Path(td) / "trial"
    report = run_direct_teacher_flat_compiler_trial(out)
    assert all(report["controls"].values()), report["controls"]
    assert report["teacher_queries"] == 16
    assert report["query_accuracy"] == 1.0
    assert report["runtime_artifact_count"] == 1
    assert report["runtime_artifacts"] == ["knowledge.eqkernel.json"]
    assert report["legacy_oracle_core_parity"] is True
    assert report["direct_compile"]["fact_count"] == 24
    assert report["direct_compile"]["concept_shortcut_count"] == 24
    assert report["direct_compile"]["route_count"] == 15
    assert report["direct_compile"]["family_alias_route_count"] == 2
    assert report["concept_macro"]["levels_built"] >= 2
    assert report["relation_macro"]["levels_built"] >= 2

    pack_path = out / "knowledge.eqkernel.json"
    pack = load_unified_numeric_kernel_pack(pack_path)
    assert pack.digest == report["direct_compile"]["digest"]

    # Deployment needs only the v0.55 execution kernel module, not the v0.56 compiler.
    code = r'''
import json, sys
from equinox_forge.unified_numeric_kernel import load_unified_numeric_kernel_pack, UnifiedNumericExecutionKernel
p = load_unified_numeric_kernel_pack(sys.argv[1])
k = UnifiedNumericExecutionKernel(p)
forbidden = [m for m in sys.modules if m in {
    "equinox_forge.direct_teacher_flat_compiler",
    "equinox_forge.numeric_runtime",
    "equinox_forge.macro_numeric_runtime",
    "equinox_forge.canonical_relation_facets",
    "equinox_forge.relation_macro_runtime",
    "equinox_forge.unified_relation_runtime",
}]
subjects = sorted({f.subject_id for f in p.facts})
r = p.routes[0]
res = k.resolve(subjects[0], r.query_relation_id)
print(json.dumps({"forbidden": forbidden, "status": res.status, "routes": len(p.routes)}))
'''
    proc = subprocess.run([sys.executable, "-c", code, str(pack_path)], check=True, capture_output=True, text=True)
    isolated = json.loads(proc.stdout)
    assert isolated["forbidden"] == []
    assert isolated["routes"] == 15

print("FORGE_V056_DIRECT_TEACHER_FLAT_COMPILER_OK")
