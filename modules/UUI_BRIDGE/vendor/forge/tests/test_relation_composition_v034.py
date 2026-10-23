from pathlib import Path
import json
import re
import tempfile

ROOT = Path(__file__).resolve().parents[1]

from equinox_forge.relation_composition import run_relation_composition_benchmark

pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
init_text = (ROOT / "equinox_forge" / "__init__.py").read_text(encoding="utf-8")
readme = (ROOT / "README.md").read_text(encoding="utf-8")
contract = json.loads((ROOT / "benchmarks" / "v0_34" / "relation_composition_contract.json").read_text(encoding="utf-8"))

project_version = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, flags=re.M).group(1)
init_version = re.search(r'__version__\s*=\s*"([^"]+)"', init_text).group(1)
assert tuple(map(int, project_version.split('.'))) >= (0, 34, 0)
assert init_version == project_version
assert readme.startswith("# Equinox Forge v0.")
assert 'eqforge-relation-compose = "equinox_forge.relation_composition:main"' in pyproject
assert contract["frozen_l5_prompt"] == "Hund wohnt in"
assert contract["expected"] == "Haus"

with tempfile.TemporaryDirectory() as td:
    out = Path(td) / "v034"
    report = run_relation_composition_benchmark(out)
    h = report.headline_metrics

    assert h["l0_exact_replay_top1"] == 1.0
    assert h["l1_suffix_transfer_top1"] == 1.0
    assert h["l2_recombination_top1"] == 1.0
    assert h["l3_novel_order_top1"] == 1.0
    assert h["l4_evidence_backed_novel_entity_top1"] == 1.0
    assert h["l5_composed_heldout_prediction_family_top1"] == 1.0

    # Necessary-condition controls: neither half of the rule is sufficient alone.
    assert h["opaque_l5_top1"] == 0.0
    assert h["graph_only_l5_top1"] == 0.0
    assert h["schema_only_l5_top1"] == 0.0
    assert h["runtime_ablation_l5_top1"] == 0.0

    assert h["composed_hund_is_haus"] is True
    assert h["stein_without_complete_path_abstains"] is True
    assert h["unmatched_relation_phrase_abstains"] is True

    positive = report.cases["composed_hund"]
    assert positive["top1"] == "Haus"
    assert positive["composition_matches"] == 1
    assert positive["composition_admissions"] == 1
    assert positive["relation_composition"]
    terminal = positive["relation_composition"][0]["terminals"][0]
    assert terminal["target_surface"] == "Haus"
    assert terminal["path_quality"] >= 0.5
    assert [x["relation_type"] for x in terminal["path_edges"]] == ["is_a", "property"]

    negative = report.cases["composed_stein"]
    assert negative["top1"] is None
    assert negative["composition_matches"] == 1
    assert negative["composition_rejections"] == 1

    assert report.controls["direct_prediction_edge_for_hund_wohnt_in"] is False
    assert report.controls["evaluation_teacher_predictions_compiled"] is False
    assert (out / "relation-composition-report.json").exists()

print("FORGE_V34_RELATION_COMPOSITION_OK")
