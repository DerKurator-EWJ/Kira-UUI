from __future__ import annotations

from pathlib import Path
import json
import tempfile

from equinox_forge.relation_macro_runtime import (
    RELATION_MACRO_RUNTIME_VERSION,
    load_relation_macro_runtime_pack,
    run_relation_macro_runtime_demo,
)

assert RELATION_MACRO_RUNTIME_VERSION == "0.53"

with tempfile.TemporaryDirectory() as td:
    out = Path(td) / "trial"
    report = run_relation_macro_runtime_demo(out)
    assert all(report["controls"].values()), report["controls"]
    assert report["relation_macro_levels"] == 2
    assert report["relation_macro_count"] == 3
    assert report["property_query"]["candidate_facets_examined"] == 4
    assert report["action_query"]["candidate_facets_examined"] == 4
    assert report["root_query"]["candidate_facets_examined"] == 8
    assert report["work"]["candidate_reduction_vs_unindexed_scan"] == 0.5
    assert report["work"]["macro_beats_specialized_family_index"] is False
    assert report["work"]["macro_matches_specialized_family_index_candidate_count"] is True
    assert len(report["property_query"]["values"]) == 3
    assert len(report["action_query"]["values"]) == 3
    assert len(report["root_query"]["values"]) == 6

    pack_path = out / "relation-macro.eqnum.json"
    pack = load_relation_macro_runtime_pack(pack_path)
    assert pack.digest == report["pack_digest"]

    # Runtime packs are integrity protected.
    payload = json.loads(pack_path.read_text(encoding="utf-8"))
    payload["metadata"]["tampered"] = True
    bad = out / "tampered.eqnum.json"
    bad.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    try:
        load_relation_macro_runtime_pack(bad)
    except ValueError as exc:
        assert "digest mismatch" in str(exc)
    else:
        raise AssertionError("tampered relation-macro runtime pack was accepted")

print("FORGE_V053_RELATION_MACRO_RUNTIME_OK")
