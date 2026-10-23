from __future__ import annotations

from pathlib import Path
import json
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from equinox_forge.macro_numeric_runtime import (
    MACRO_NUMERIC_RUNTIME_FORMAT,
    MACRO_NUMERIC_RUNTIME_VERSION,
    load_macro_numeric_runtime_pack,
    run_macro_numeric_runtime_demo,
)

assert MACRO_NUMERIC_RUNTIME_VERSION == "0.49"

with tempfile.TemporaryDirectory() as td:
    out = Path(td) / "demo"
    report = run_macro_numeric_runtime_demo(out)
    assert all(report["controls"].values()), report["controls"]
    assert report["work"]["baseline_nodes_touched"] == 38
    assert report["work"]["macro_nodes_touched"] == 23
    assert report["work"]["macro_routed_queries"] == 11
    assert report["work"]["node_reduction_fraction"] > 0.39
    assert report["shortcut_count"] >= 20
    assert report["macro_levels"] == 2
    assert report["macro_count"] == 3

    pack_path = out / "macro.eqnum.json"
    pack = load_macro_numeric_runtime_pack(pack_path)
    assert pack.format == MACRO_NUMERIC_RUNTIME_FORMAT
    assert pack.version == MACRO_NUMERIC_RUNTIME_VERSION
    assert pack.digest == report["pack_digest"]

    # Digest corruption must be rejected.
    payload = json.loads(pack_path.read_text(encoding="utf-8"))
    payload["shortcuts"][0][3] += 1
    bad = out / "tampered.eqnum.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    try:
        load_macro_numeric_runtime_pack(bad)
    except ValueError as exc:
        assert "digest mismatch" in str(exc)
    else:
        raise AssertionError("tampered macro runtime pack must fail digest validation")

print("FORGE_V049_MACRO_NUMERIC_RUNTIME_OK")
