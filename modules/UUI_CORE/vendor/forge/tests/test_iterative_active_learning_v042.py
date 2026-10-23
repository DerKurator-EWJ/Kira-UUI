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
    print("FORGE_V042_ITERATIVE_ACTIVE_SKIPPED_NO_TORCH")
    raise SystemExit(0)

from equinox_forge.iterative_active_learning import run_iterative_active_trial

pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
init_text = (ROOT / "equinox_forge" / "__init__.py").read_text(encoding="utf-8")
contract = json.loads((ROOT / "benchmarks" / "v0_42" / "iterative_active_contract.json").read_text(encoding="utf-8"))
reference = json.loads((ROOT / "benchmarks" / "v0_42" / "reference.json").read_text(encoding="utf-8"))
project_version = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, flags=re.M).group(1)
init_version = re.search(r'__version__\s*=\s*"([^"]+)"', init_text).group(1)
assert init_version == project_version
assert 'eqforge-iterative-active = "equinox_forge.iterative_active_learning:main"' in pyproject
assert contract["runtime_changed"] is False

# Frozen full-domain audit: iteration currently does not change the final active
# schedule or improve the strategy-independent shadow holdout.
assert reference["findings"]["iterative_final_equals_one_shot_final_selection"] is True
assert reference["findings"]["genuine_fixed_holdout_transfer_improvement_observed"] is False
assert reference["iterative_curve"][-1]["full_accuracy"] == 0.6171875
assert reference["fixed_curve"][-1]["full_accuracy"] == 0.5703125
assert reference["iterative_curve"][-1]["shadow_holdout_accuracy"] == 0.15625
assert reference["fixed_curve"][-1]["shadow_holdout_accuracy"] == 0.15625
assert reference["random_curve"][-1]["shadow_holdout_accuracy"]["mean"] == 0.15625

# Smaller neural smoke run validates invariants without paying the full benchmark
# cost during every regression suite execution.
with tempfile.TemporaryDirectory() as td:
    report = run_iterative_active_trial(
        Path(td) / "trial",
        max_entities=8,
        final_budget=16,
        batch_size=4,
        random_trials=2,
        seed=3901,
        random_seed=4210,
    )
    assert report["runtime_cognition_changed_in_v042"] is False
    assert report["selector_uses_teacher_answer_before_selection"] is False
    assert report["selector_uses_expected_labels"] is False
    assert report["shadow_holdout_facts"] == 8
    assert report["selectable_facts"] == 24
    assert report["controls"]["shadow_never_selectable"] is True
    assert report["controls"]["seed_and_shadow_disjoint"] is True
    assert report["controls"]["iterative_unique_teacher_queries_equal_budget"] is True
    assert report["controls"]["one_shot_unique_teacher_queries_equal_budget"] is True
    assert report["controls"]["fixed_unique_teacher_queries_equal_budget"] is True
    assert report["controls"]["all_random_unique_teacher_queries_equal_budget"] is True
    assert report["controls"]["iterative_opaque_relation_abstention"] == 1.0
    assert report["controls"]["iterative_outside_domain_abstention"] == 1.0
    assert report["findings"]["iterative_final_equals_one_shot_final_selection"] is True
    assert report["findings"]["genuine_fixed_holdout_transfer_improvement_observed"] is False

print("FORGE_V042_ITERATIVE_ACTIVE_LEARNING_OK")
