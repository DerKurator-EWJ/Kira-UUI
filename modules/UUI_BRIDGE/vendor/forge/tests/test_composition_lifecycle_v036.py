from pathlib import Path
import json
import re
import tempfile

ROOT = Path(__file__).resolve().parents[1]

from equinox_forge.composition_lifecycle import run_composition_lifecycle_benchmark

pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
init_text = (ROOT / "equinox_forge" / "__init__.py").read_text(encoding="utf-8")
contract = json.loads((ROOT / "benchmarks" / "v0_36" / "composition_lifecycle_contract.json").read_text(encoding="utf-8"))

project_version = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, flags=re.M).group(1)
init_version = re.search(r'__version__\s*=\s*"([^"]+)"', init_text).group(1)
assert tuple(map(int, project_version.split('.'))) >= (0, 36, 0)
assert init_version == project_version
assert 'eqforge-composition-lifecycle = "equinox_forge.composition_lifecycle:main"' in pyproject

with tempfile.TemporaryDirectory() as td:
    report = run_composition_lifecycle_benchmark(Path(td) / "v036")
    h = report.headline_metrics
    assert h["candidate_after_first_support"] is True
    assert h["stable_after_second_support"] is True
    assert h["contested_after_first_conflict"] is True
    assert h["quarantined_after_second_conflict"] is True
    assert h["candidate_after_first_recovery"] is True
    assert h["stable_after_second_recovery"] is True
    assert h["stable_runtime_returns_haus"] is True
    assert h["nonstable_runtime_abstains"] is True
    assert h["final_primary_status"] == "stable"

    statuses = [x["primary"]["status"] for x in report.lifecycle_runs]
    assert statuses == contract["expected_status_sequence"]
    assert report.lifecycle_runs[0]["runtime_top1"] is None
    assert report.lifecycle_runs[1]["runtime_top1"] == "Haus"
    assert report.lifecycle_runs[2]["runtime_top1"] is None
    assert report.lifecycle_runs[3]["runtime_top1"] is None
    assert report.lifecycle_runs[4]["runtime_top1"] is None
    assert report.lifecycle_runs[5]["runtime_top1"] == "Haus"

    assert all(report.controls.values())

print("FORGE_V36_COMPOSITION_LIFECYCLE_OK")
