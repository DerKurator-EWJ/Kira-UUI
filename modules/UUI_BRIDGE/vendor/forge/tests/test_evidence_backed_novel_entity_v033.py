from pathlib import Path
import json
import tempfile
import re

ROOT = Path(__file__).resolve().parents[1]

from equinox_forge.novel_entity_admission import run_novel_entity_admission_benchmark

pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
init_text = (ROOT / "equinox_forge" / "__init__.py").read_text(encoding="utf-8")
readme = (ROOT / "README.md").read_text(encoding="utf-8")
contract = json.loads((ROOT / "benchmarks" / "v0_33" / "novel_entity_contract.json").read_text(encoding="utf-8"))

project_version = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, flags=re.M).group(1)
init_version = re.search(r'__version__\s*=\s*"([^"]+)"', init_text).group(1)
assert tuple(map(int, project_version.split('.'))) >= (0, 33, 0)
assert init_version == project_version
assert readme.startswith("# Equinox Forge v0.")
assert 'eqforge-novel-entity = "equinox_forge.novel_entity_admission:main"' in pyproject
assert contract["positive_case"]["prompt"] == "Wolf ist ein"
assert contract["negative_case"]["prompt"] == "Stein ist ein"

with tempfile.TemporaryDirectory() as td:
    report = run_novel_entity_admission_benchmark(Path(td) / "v033")
    h = report.headline_metrics
    assert h["l0_exact_replay_top1"] == 1.0
    assert h["l1_suffix_transfer_top1"] == 1.0
    assert h["l2_recombination_top1"] == 1.0
    assert h["l3_novel_order_top1"] == 1.0
    assert h["l4_evidence_backed_novel_entity_top1"] == 1.0
    assert h["l5_heldout_relation_top1"] == 0.0
    assert h["surface_only_l4_top1"] == 0.0
    assert h["wolf_semantic_admitted"] is True
    assert h["stone_semantic_rejected"] is True
    assert h["surface_presence_alone_rejected"] is True
    assert h["known_recombination_preserved"] is True
    assert h["unsupported_known_mix_rejected"] is True

    cases = report.cases
    assert cases["semantic_wolf"]["top1"] == "Tier"
    assert cases["semantic_wolf"]["semantic_admissions"] >= 1
    assert cases["semantic_stone"]["top1"] is None
    assert cases["semantic_stone"]["semantic_rejections"] >= 1
    assert cases["surface_only_wolf"]["top1"] is None
    assert cases["surface_only_wolf"]["semantic_rejections"] >= 1
    assert (Path(td) / "v033" / "novel-entity-admission-report.json").exists()

print("FORGE_V33_EVIDENCE_BACKED_NOVEL_ENTITY_OK")
