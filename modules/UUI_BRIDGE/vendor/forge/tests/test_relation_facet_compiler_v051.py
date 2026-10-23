from __future__ import annotations

from pathlib import Path
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from equinox_forge.relation_facet_compiler import (
    RELATION_FACET_COMPILER_VERSION,
    run_relation_facet_compiler_trial,
)

assert RELATION_FACET_COMPILER_VERSION == "0.51"

with tempfile.TemporaryDirectory() as td:
    report = run_relation_facet_compiler_trial(Path(td) / "trial")
    assert all(report["controls"].values()), report["controls"]
    assert report["family_query_accuracy"] == 1.0
    assert report["relation_facet_count"] == 8
    assert report["relation_hierarchy_triple_count"] == 8
    assert report["work"]["macro_shortcut_count"] > 0
    assert report["work"]["macro_routed_values"] > 0
    assert report["work"]["node_reduction_fraction"] > 0.0
    assert report["teacher_disabled_before_runtime"] is True
    assert report["forge_db_removed_before_runtime"] is True

print("FORGE_V051_RELATION_FACET_COMPILER_OK")
