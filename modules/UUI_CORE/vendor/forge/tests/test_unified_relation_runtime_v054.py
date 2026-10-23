from __future__ import annotations

from pathlib import Path
import json
import tempfile

from equinox_forge.unified_relation_runtime import (
    UNIFIED_RELATION_RUNTIME_VERSION,
    load_unified_relation_runtime_pack,
    run_unified_relation_runtime_demo,
)

assert UNIFIED_RELATION_RUNTIME_VERSION == "0.54"

with tempfile.TemporaryDirectory() as td:
    out = Path(td) / "trial"
    report = run_unified_relation_runtime_demo(out)
    assert all(report["controls"].values()), report["controls"]
    assert report["family_alias_route_count"] == 2
    assert report["direct_macro_route_count"] == 3
    assert report["property_query"]["route_kind"] == "relation_macro"
    assert report["action_query"]["route_kind"] == "relation_macro"
    assert report["scalar_query"]["route_kind"] == "scalar"
    assert len(report["property_query"]["values"]) == 3
    assert len(report["action_query"]["values"]) == 3
    assert len(report["root_macro_query"]["values"]) == 6

    pack_path = out / "unified-relation.eqnum.json"
    pack = load_unified_relation_runtime_pack(pack_path)
    assert pack.digest == report["pack_digest"]

    payload = json.loads(pack_path.read_text(encoding="utf-8"))
    payload["routes"][0]["execution_relation_id"] += 1
    bad = out / "tampered.eqnum.json"
    bad.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    try:
        load_unified_relation_runtime_pack(bad)
    except ValueError as exc:
        assert "digest mismatch" in str(exc)
    else:
        raise AssertionError("tampered unified relation runtime pack was accepted")

print("FORGE_V054_UNIFIED_RELATION_RUNTIME_OK")
