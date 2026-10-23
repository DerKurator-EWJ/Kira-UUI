from pathlib import Path
import json
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from equinox_forge.support_gate_benchmark import run_support_gate_benchmark
from equinox_forge.runtime_cognition import RuntimeCognition, RuntimeCognitionPolicy
from equinox_forge.runtime_pack import load_base_pack

pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
init_text = (ROOT / "equinox_forge" / "__init__.py").read_text(encoding="utf-8")
readme = (ROOT / "README.md").read_text(encoding="utf-8")
contract = json.loads((ROOT / "benchmarks" / "v0_32" / "support_gate_contract.json").read_text(encoding="utf-8"))
reference = json.loads((ROOT / "benchmarks" / "v0_32" / "reference.json").read_text(encoding="utf-8"))

import re
project_version = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, flags=re.M).group(1)
init_version = re.search(r'__version__\s*=\s*"([^"]+)"', init_text).group(1)
assert tuple(map(int, project_version.split('.'))) >= (0, 32, 0)
assert init_version == project_version
assert readme.startswith("# Equinox Forge v0.")
assert 'eqforge-support-gate = "equinox_forge.support_gate_benchmark:main"' in pyproject
assert contract["dataset_sha256"] == reference["dataset_sha256"]

with tempfile.TemporaryDirectory() as td:
    out = Path(td) / "support_gate"
    report = run_support_gate_benchmark(out)
    h = report.headline_metrics
    assert report.dataset_sha256 == reference["dataset_sha256"]
    assert h["exact_replay_top1"] == 1.0
    assert h["known_suffix_transfer_top1"] == 1.0
    assert h["known_parts_recombination_top1"] == 1.0
    assert h["novel_order_known_lexicon_top1"] == 1.0
    assert h["novel_entity_known_frame_top1"] == 0.0
    assert h["heldout_relation_family_top1"] == 0.0
    assert h["unknown_rejection_rate"] == 1.0
    assert h["known_or_mixed_unsupported_rejection_rate"] == 1.0
    assert h["typed_frame_precision"] == 1.0
    assert h["typed_frame_recall"] == 1.0
    assert report.delta_vs_v031["unknown_rejection_rate"] == 0.75
    assert report.delta_vs_v031["known_or_mixed_unsupported_rejection_rate"] == 1.0
    assert report.delta_vs_v031["novel_entity_known_frame_top1"] == -1.0

    pair = report.indistinguishability_case
    assert pair["legacy"]["wolf"]["top1"] == "Tier"
    assert pair["legacy"]["stone"]["top1"] == "Tier"
    assert pair["gated"]["wolf"]["top1"] is None
    assert pair["gated"]["stone"]["top1"] is None

    pack = load_base_pack(out / "01_generalization_and_rejection" / "ladder" / "train.eqbase")
    gated = RuntimeCognition(pack, RuntimeCognitionPolicy(
        max_recall_nodes=96, max_recall_depth=3, max_predictions=3, min_prediction_score=0.0,
    ))
    positive = gated.predict("Hund jagt Maus")
    negative = gated.predict("Hund Baum")
    assert positive.predictions and positive.predictions[0].surface == "heute"
    assert positive.work["support_gate_admissions"] >= 1
    assert not negative.predictions
    assert negative.work["support_gate_rejections"] >= 1
    assert negative.work["support_gate_reason:unknown_live_token_without_compiled_context"] >= 1
    assert (out / "support-gate-report.json").exists()

print("FORGE_V32_SUPPORT_GATED_PREDICTION_OK")
