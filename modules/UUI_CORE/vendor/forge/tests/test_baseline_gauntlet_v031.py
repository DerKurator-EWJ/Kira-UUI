from pathlib import Path
import json
import tempfile

from equinox_forge.baseline_gauntlet import GAUNTLET_DATA, gauntlet_dataset_sha256, run_baseline_gauntlet

ROOT = Path(__file__).resolve().parents[1]

# Frozen source-of-truth cases must match the auditable repository snapshot.
snapshot = json.loads((ROOT / "benchmarks" / "v0_31" / "gauntlet_cases.json").read_text(encoding="utf-8"))
assert snapshot == GAUNTLET_DATA
reference = json.loads((ROOT / "benchmarks" / "v0_31" / "baseline_reference.json").read_text(encoding="utf-8"))
assert reference["dataset_sha256"] == gauntlet_dataset_sha256()

# Live release metadata must identify this milestone.
pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
init_text = (ROOT / "equinox_forge" / "__init__.py").read_text(encoding="utf-8")
readme = (ROOT / "README.md").read_text(encoding="utf-8")
import re
project_version = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, flags=re.M).group(1)
init_version = re.search(r'__version__\s*=\s*"([^"]+)"', init_text).group(1)
assert tuple(map(int, project_version.split('.'))) >= (0, 31, 0)
assert init_version == project_version
assert readme.startswith("# Equinox Forge v0.")
assert 'eqforge-gauntlet = "equinox_forge.baseline_gauntlet:main"' in pyproject

with tempfile.TemporaryDirectory() as td:
    report = run_baseline_gauntlet(Path(td) / "gauntlet")
    h = report.headline_metrics
    assert report.dataset_sha256 == reference["dataset_sha256"]
    assert h["exact_replay_top1"] == 1.0
    assert h["known_suffix_transfer_top1"] == 1.0
    assert h["known_parts_recombination_top1"] == 1.0
    assert h["novel_order_known_lexicon_top1"] == 1.0
    assert h["novel_entity_known_frame_top1"] == 1.0
    assert h["heldout_relation_family_top1"] == 0.0
    assert h["unknown_rejection_rate"] == 0.25
    assert h["known_or_mixed_unsupported_rejection_rate"] == 0.0
    assert h["typed_frame_precision"] == 1.0
    assert h["typed_frame_recall"] == 1.0
    assert h["ambiguity_cluster_purity"] == 1.0
    assert h["exception_lifecycle_contract"] is True
    assert h["stable_exception_transfer_top1"] == "Sauerstoff"
    assert abs(h["runtime_macro_route_work_reduction"] - 0.746875) < 1e-12
    assert len(report.exposed_failures) == 3
    assert (Path(td) / "gauntlet" / "baseline-gauntlet-report.json").exists()

print("FORGE_V31_BASELINE_GAUNTLET_OK")
