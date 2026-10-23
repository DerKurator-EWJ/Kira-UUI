from __future__ import annotations

from pathlib import Path
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from equinox_forge.teacher_numeric_compiler import (
    TEACHER_NUMERIC_COMPILER_VERSION,
    run_teacher_numeric_compiler_trial,
)

assert TEACHER_NUMERIC_COMPILER_VERSION == "0.50"

with tempfile.TemporaryDirectory() as td:
    report = run_teacher_numeric_compiler_trial(Path(td) / "trial")
    assert all(report["controls"].values()), report["controls"]
    assert report["stored_teacher_fact_retention"] == 1.0
    assert report["scalar_runtime_query_accuracy"] == 1.0
    assert report["teacher_disabled_before_runtime"] is True
    assert report["macro"]["levels_built"] >= 2
    assert report["macro"]["macro_count"] >= 3
    assert report["work"]["macro_routed_queries"] == 0
    assert report["work"]["macro_shortcut_count"] == 0
    assert report["work"]["node_reduction_fraction"] == 0.0
    assert report["known_limitations"]["generic_property_action_are_scalar_queries"] is True
    assert report["known_limitations"]["full_relational_storage_is_preserved_despite_scalar_query_limit"] is True
    assert report["known_limitations"]["coarse_predicates_block_safe_macro_shortcuts"] is True
    assert report["external_pretrained_hf_claim"] is False

print("FORGE_V050_TEACHER_NUMERIC_COMPILER_OK")
