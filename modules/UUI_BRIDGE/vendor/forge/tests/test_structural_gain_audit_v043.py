from __future__ import annotations

import json
from pathlib import Path
import re
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch  # noqa: F401
except Exception:
    print("FORGE_V043_STRUCTURAL_GAIN_SKIPPED_NO_TORCH")
    raise SystemExit(0)

from equinox_forge.structural_gain_audit import run_structural_gain_audit

pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
init_text = (ROOT / "equinox_forge" / "__init__.py").read_text(encoding="utf-8")
contract = json.loads((ROOT / "benchmarks" / "v0_43" / "structural_gain_contract.json").read_text(encoding="utf-8"))
reference = json.loads((ROOT / "benchmarks" / "v0_43" / "reference.json").read_text(encoding="utf-8"))
project_version = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, flags=re.M).group(1)
init_version = re.search(r'__version__\s*=\s*"([^"]+)"', init_text).group(1)
assert tuple(map(int, project_version.split("."))) >= (0, 43, 0)
assert init_version == project_version
assert (ROOT / "RELEASE_V0_43.md").exists()
assert 'eqforge-structural-gain-audit = "equinox_forge.structural_gain_audit:main"' in pyproject
assert contract["runtime_changed"] is False

assert reference["seed"]["shadow_accuracy"] == 0.15625
assert reference["active_budgeted"]["shadow_accuracy"] == 0.15625
assert reference["all_selectable_saturation"]["shadow_accuracy"] == 0.15625
assert reference["all_selectable_saturation"]["full_accuracy"] == 0.7890625
assert reference["diffs"]["saturation_vs_seed"]["top1_changes"] == 0
assert reference["diffs"]["saturation_vs_seed"]["score_changes"] == 16
assert reference["random_schedule_search"]["shadow_accuracy_min"] == 0.15625
assert reference["random_schedule_search"]["shadow_accuracy_max"] == 0.15625
assert reference["findings"]["selector_leverage_observed_on_frozen_shadow"] is False

with tempfile.TemporaryDirectory() as td:
    report = run_structural_gain_audit(
        Path(td) / "trial",
        max_entities=8,
        query_budget=16,
        random_trials=3,
        seed=3901,
        random_seed=4301,
    )
    assert report["runtime_cognition_changed_in_v043"] is False
    assert report["shadow_holdout_facts"] == 8
    assert report["controls"]["shadow_never_queried_in_seed"] is True
    assert report["controls"]["shadow_never_queried_in_active"] is True
    assert report["controls"]["shadow_never_queried_in_saturation"] is True
    assert report["controls"]["opaque_relation_abstention_active"] == 1.0
    assert report["controls"]["outside_domain_abstention_active"] == 1.0
    assert report["controls"]["opaque_relation_abstention_saturation"] == 1.0
    assert report["controls"]["outside_domain_abstention_saturation"] == 1.0

print("FORGE_V043_STRUCTURAL_GAIN_AUDIT_OK")
