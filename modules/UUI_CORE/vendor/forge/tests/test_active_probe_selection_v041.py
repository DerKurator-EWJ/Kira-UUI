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
    print("FORGE_V041_ACTIVE_PROBE_SKIPPED_NO_TORCH")
    raise SystemExit(0)

from equinox_forge.active_probe_selection import run_active_probe_trial

pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
init_text = (ROOT / "equinox_forge" / "__init__.py").read_text(encoding="utf-8")
contract = json.loads((ROOT / "benchmarks" / "v0_41" / "active_probe_contract.json").read_text(encoding="utf-8"))
reference = json.loads((ROOT / "benchmarks" / "v0_41" / "reference.json").read_text(encoding="utf-8"))
project_version = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, flags=re.M).group(1)
init_version = re.search(r'__version__\s*=\s*"([^"]+)"', init_text).group(1)
assert init_version == project_version
assert tuple(int(x) for x in project_version.split(".")) >= (0, 41, 0)
assert 'eqforge-active-probe = "equinox_forge.active_probe_selection:main"' in pyproject
assert contract["runtime_changed"] is False
assert reference["active"]["full_accuracy"] > reference["fixed"]["full_accuracy"]
assert reference["active"]["unqueried_accuracy"] > reference["fixed"]["unqueried_accuracy"]

# Smaller neural smoke trial keeps regression cost bounded while verifying the
# acquisition/query-accounting invariants end to end.
with tempfile.TemporaryDirectory() as td:
    report = run_active_probe_trial(
        Path(td) / "trial",
        max_entities=8,
        query_budget=16,
        random_trials=2,
        seed=3901,
        random_seed=4110,
    )
    assert report["selector_uses_teacher_answer_before_selection"] is False
    assert report["selector_uses_expected_labels"] is False
    assert report["runtime_cognition_changed_in_v041"] is False
    assert report["active"]["teacher_queries"] == 16
    assert report["fixed"]["teacher_queries"] == 16
    assert report["controls"]["all_random_unique_teacher_queries_equal_budget"] is True
    assert report["active"]["teacher_cache_hits"] >= 8
    assert report["active"]["evaluation"]["opaque_relation_abstention_rate"] == 1.0
    assert report["active"]["evaluation"]["outside_domain_abstention_rate"] == 1.0
    # The selector diagnostics expose only prompt/runtime uncertainty metadata,
    # never the hidden benchmark target.
    for row in report["active"]["frontier_diagnostics"]:
        assert "expected" not in row
        assert "teacher_top1" not in row

print("FORGE_V041_ACTIVE_PROBE_SELECTION_OK")
