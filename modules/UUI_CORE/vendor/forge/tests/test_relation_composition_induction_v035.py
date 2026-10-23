from pathlib import Path
import json
import re
import tempfile

ROOT = Path(__file__).resolve().parents[1]

from equinox_forge.relation_composition_induction import run_relation_composition_induction_benchmark

pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
init_text = (ROOT / "equinox_forge" / "__init__.py").read_text(encoding="utf-8")
contract = json.loads((ROOT / "benchmarks" / "v0_35" / "relation_composition_induction_contract.json").read_text(encoding="utf-8"))

project_version = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, flags=re.M).group(1)
init_version = re.search(r'__version__\s*=\s*"([^"]+)"', init_text).group(1)
assert tuple(map(int, project_version.split('.'))) >= (0, 35, 0)
assert init_version == project_version
assert 'eqforge-relation-induce = "equinox_forge.relation_composition_induction:main"' in pyproject
assert contract["heldout_prompt"] == "Hund wohnt in"
assert contract["expected"] == "Haus"

with tempfile.TemporaryDirectory() as td:
    report = run_relation_composition_induction_benchmark(Path(td) / "v035")
    h = report.headline_metrics
    assert h["l0_exact_replay_top1"] == 1.0
    assert h["l1_suffix_transfer_top1"] == 1.0
    assert h["l2_recombination_top1"] == 1.0
    assert h["l3_novel_order_top1"] == 1.0
    assert h["l4_evidence_backed_novel_entity_top1"] == 1.0
    assert h["l5_auto_induced_relation_composition_top1"] == 1.0
    assert h["stable_schema_count"] == 1
    assert h["same_source_stable_schema_count"] == 0
    assert h["sparse_evidence_stable_schema_count"] == 0
    assert h["competing_program_stable_schema_count"] == 0
    assert h["heldout_hund_is_haus"] is True
    assert h["stein_abstains"] is True
    assert h["unmatched_phrase_abstains"] is True

    stable = [x for x in report.induction["positive"]["candidates"] if x["status"] == "stable"]
    assert len(stable) == 1
    assert stable[0]["pattern_tokens"] == ["[SLOT_0]", "wohnt", "in"]
    assert [x["relation_type"] for x in stable[0]["path"]] == ["is_a", "property"]
    assert stable[0]["support_count"] == 3
    assert stable[0]["source_count"] == 3
    assert stable[0]["coverage"] == 1.0

    assert report.controls["direct_observations_compiled_into_runtime"] is False
    assert report.controls["relation_phrase_supplied_to_inducer"] is False
    assert report.controls["heldout_hund_present_in_induction_observations"] is False
    assert report.controls["same_source_is_not_independent_consensus"] is True
    assert report.controls["incomplete_coverage_is_not_stable"] is True
    assert report.controls["equal_competing_programs_are_not_arbitrarily_selected"] is True

print("FORGE_V35_RELATION_COMPOSITION_INDUCTION_OK")
