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
    print("FORGE_V039_DOMAIN_TRIAL_SKIPPED_NO_TORCH")
    raise SystemExit(0)

from equinox_forge.real_teacher_domain_trial import run_domain_trial

pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
init_text = (ROOT / "equinox_forge" / "__init__.py").read_text(encoding="utf-8")
contract = json.loads((ROOT / "benchmarks" / "v0_39" / "domain_trial_contract.json").read_text(encoding="utf-8"))
project_version = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, flags=re.M).group(1)
init_version = re.search(r'__version__\s*=\s*"([^"]+)"', init_text).group(1)
assert tuple(map(int, project_version.split('.'))) >= (0, 39, 0)
assert init_version == project_version
assert 'eqforge-domain-trial = "equinox_forge.real_teacher_domain_trial:main"' in pyproject

with tempfile.TemporaryDirectory() as td:
    report = run_domain_trial(Path(td) / "trial", max_entities=12, seed=3901)
    assert report["teacher_is_computed_neural_model"] is True
    assert report["pretrained_external_model_claim"] is False
    assert report["teacher"]["training_accuracy"] >= contract["required_invariants"]["teacher_training_accuracy_min"]
    assert report["replay_compiler_report"]["direct_replay"]["top1_retention"] >= contract["required_invariants"]["direct_replay_top1_retention_min"]
    assert report["outside_domain_abstention_rate"] >= contract["required_invariants"]["outside_domain_abstention_min"]
    assert report["categories"]["opaque_relation"]["runtime_abstention_rate"] == 1.0
    assert report["compiled_probe_facts"] == 24
    assert report["uncompiled_core_facts"] == 24
    assert report["opaque_relation_facts"] == 12
    assert report["pack_bytes"] > 0
    assert report["teacher_state_bytes"] > 0

print("FORGE_V039_REAL_TEACHER_DOMAIN_TRIAL_OK")
