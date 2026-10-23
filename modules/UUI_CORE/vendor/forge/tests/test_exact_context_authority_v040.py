from pathlib import Path
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from equinox_forge.database import CanonicalDB
from equinox_forge.ir import IRNode, RelationEvidence, SourceRef, NodeKind, Modality, RelationType
from equinox_forge.macro_discovery import MacroDiscoveryPolicy
from equinox_forge.runtime_compression import MacroPromotionPolicy, compile_runtime_compression
from equinox_forge.runtime_pack import compile_base_pack, load_base_pack
from equinox_forge.runtime_cognition import RuntimeCognition, RuntimeCognitionPolicy



import json
import re

pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
init_text = (ROOT / "equinox_forge" / "__init__.py").read_text(encoding="utf-8")
reference = json.loads((ROOT / "benchmarks" / "v0_40" / "reference.json").read_text(encoding="utf-8"))
contract = json.loads((ROOT / "benchmarks" / "v0_40" / "exact_context_authority_contract.json").read_text(encoding="utf-8"))
project_version = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, flags=re.M).group(1)
init_version = re.search(r'__version__\s*=\s*"([^"]+)"', init_text).group(1)
assert tuple(map(int, project_version.split('.'))) >= (0, 40, 0)
assert init_version == project_version
assert 'eqforge-exact-authority = "equinox_forge.exact_context_authority:main"' in pyproject
assert contract["treatment"]["same_eqbase"] is True
assert reference["v040"]["replay"]["accuracy"] == 1.0
assert reference["v040"]["paraphrase"]["accuracy"] == 1.0
assert reference["v040"]["heldout_fact"]["accuracy"] == reference["baseline_v039"]["heldout_fact"]["accuracy"]

def node(node_id, kind, surface, members=None, level=0):
    return IRNode(node_id, Modality.TEXT.value, kind, level, surface, list(members or []), {})


with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    db_path = root / "authority.sqlite"
    pack_path = root / "authority.eqbase"

    tokens = ["löwe", "hund", "kuh", "pferd", "ziege", "schaf", "fuchs", "hase", "maus", "katze", "lebt", "im", "savanne", "land"]
    nodes = [node(f"s:{x}", NodeKind.SURFACE.value, x) for x in tokens]
    live = node("ctx:lion", NodeKind.TEMPLATE.value, "löwe lebt im", ["s:löwe", "s:lebt", "s:im"], level=1)
    nodes.append(live)
    others = ["hund", "kuh", "pferd", "ziege", "schaf", "fuchs", "hase", "maus", "katze"]
    for x in others:
        nodes.append(node(f"ctx:{x}", NodeKind.TEMPLATE.value, f"{x} lebt im", [f"s:{x}", "s:lebt", "s:im"], level=1))

    src = SourceRef("fixture", "1", "v040-authority")
    evidence = [
        RelationEvidence("ctx:lion", "s:savanne", RelationType.PREDICTION.value, 0.90, 0.98, src, {}),
    ]
    for x in others:
        evidence.append(RelationEvidence(f"ctx:{x}", "s:land", RelationType.PREDICTION.value, 0.90, 0.98, src, {}))

    with CanonicalDB(db_path) as c:
        for n in nodes:
            c.upsert_node(n)
        for e in evidence:
            c.add_evidence(e)
        c.commit()
        compression = compile_runtime_compression(
            c.db,
            MacroDiscoveryPolicy(min_pattern_size=2, max_pattern_size=4, min_occurrences=999, min_parent_structures=999, definition_overhead=1.0, min_net_savings=1.0, min_compression_ratio=0.0, prune_dominated=True),
            MacroPromotionPolicy(min_realized_occurrences=999, min_realized_parent_structures=999, definition_overhead=1.0, min_realized_net_savings=1.0, min_realized_compression_ratio=0.0, max_promotions=1),
            persist=True, materialize=True,
        )
        compile_base_pack(c.db, pack_path, pack_id="v040-authority", pack_version=40, compression_run_id=compression.compression_run_id, include_metadata=True, persist=False, verify_roundtrip=True)

    pack = load_base_pack(pack_path)
    common = dict(max_recall_nodes=96, max_recall_depth=3, max_predictions=4, min_prediction_score=0.0)

    legacy = RuntimeCognition(pack, RuntimeCognitionPolicy(**common, enable_exact_context_authority=False))
    old = legacy.predict("löwe lebt im")
    assert old.predictions and old.predictions[0].surface == "land", old.to_dict()

    gated = RuntimeCognition(pack, RuntimeCognitionPolicy(**common, enable_exact_context_authority=True, exact_context_authority_min_direct_score=0.50))
    new = gated.predict("löwe lebt im")
    assert new.predictions and new.predictions[0].surface == "savanne", new.to_dict()
    assert new.work["exact_context_authority_active"] == 1
    assert new.predictions[0].direct_score >= 0.50
    land = next(p for p in new.predictions if p.surface == "land")
    assert land.score > new.predictions[0].score, "authority must beat aggregation by evidence class, not by hidden score inflation"

    # This is deliberately not an unconditional exact-wins rule. Raising the
    # quality floor above the direct evidence disables the authority tier and
    # restores ordinary aggregation over the exact same pack and query.
    too_strict = RuntimeCognition(pack, RuntimeCognitionPolicy(**common, enable_exact_context_authority=True, exact_context_authority_min_direct_score=0.95))
    weak = too_strict.predict("löwe lebt im")
    assert weak.work["exact_context_authority_active"] == 0
    assert weak.predictions and weak.predictions[0].surface == "land", weak.to_dict()

print("FORGE_V40_EXACT_CONTEXT_AUTHORITY_OK")
