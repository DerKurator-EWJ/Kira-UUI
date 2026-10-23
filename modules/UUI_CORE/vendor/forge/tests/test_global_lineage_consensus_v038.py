from pathlib import Path
import json
import re
import sqlite3
import tempfile

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from equinox_forge.compiler import CompilePolicy, aggregate_relations
from equinox_forge.consensus import SourceVote, _collapse_votes_by_lineage
from equinox_forge.database import CanonicalDB
from equinox_forge.global_lineage_consensus import run_global_lineage_consensus_benchmark
from equinox_forge.ir import RelationEvidence, RelationType, SourceRef
from equinox_forge.canonical import make_text_surface
from equinox_forge.source_lineage import register_source_lineage

pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
init_text = (ROOT / "equinox_forge" / "__init__.py").read_text(encoding="utf-8")
contract = json.loads((ROOT / "benchmarks" / "v0_38" / "global_lineage_consensus_contract.json").read_text(encoding="utf-8"))
project_version = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, flags=re.M).group(1)
init_version = re.search(r'__version__\s*=\s*"([^"]+)"', init_text).group(1)
assert tuple(map(int, project_version.split('.'))) >= (0, 38, 0)
assert init_version == project_version
assert 'eqforge-global-lineage = "equinox_forge.global_lineage_consensus:main"' in pyproject

# Direct adversarial vote check: support and conflict inside one family must abstain.
db = sqlite3.connect(":memory:")
register_source_lineage(db, "sib-a", "1", "root-x", "1")
register_source_lineage(db, "sib-b", "1", "root-x", "1")
votes = [
    SourceVote("sib-a@1", 1.0, "support", .9, .1, 1, 1, {}, .8),
    SourceVote("sib-b@1", 1.0, "conflict", .1, .9, 1, 1, {}, .8),
]
collapsed = _collapse_votes_by_lineage(db, votes)
assert len(collapsed) == 1
assert collapsed[0].vote == "abstain"
assert collapsed[0].lineage_internal_conflict is True
assert set(collapsed[0].raw_source_keys) == {"sib-a@1", "sib-b@1"}

# Relation score itself must be alias-count invariant inside one lineage.
with tempfile.TemporaryDirectory() as td:
    with CanonicalDB(Path(td) / "alias.sqlite") as cdb:
        qa, a = make_text_surface("A")
        qb, b = make_text_surface("B")
        for n in [*qa, a, *qb, b]: cdb.upsert_node(n)
        for child in ("low", "hi-1", "hi-2", "hi-3"):
            register_source_lineage(cdb.db, child, "1", "root-z", "1")
        vals = {"low": .1, "hi-1": .9, "hi-2": .9, "hi-3": .9}
        for src, score in vals.items():
            cdb.add_evidence(RelationEvidence(a.node_id, b.node_id, RelationType.ASSOCIATION.value, score, .8, SourceRef(src, "1", "v038_alias"), {}))
        cdb.commit()
        rels = aggregate_relations(cdb.db, CompilePolicy(min_score=0, min_confidence=0, min_sources=1))
        rel = next(r for r in rels if r.from_id == a.node_id and r.to_id == b.node_id)
        assert rel.source_count == 1
        assert abs(rel.score - .5) < 1e-12, rel
        assert abs(rel.confidence - .8) < 1e-12, rel

with tempfile.TemporaryDirectory() as td:
    report = run_global_lineage_consensus_benchmark(Path(td) / "v038")
    h = report.headline_metrics
    assert h["relation_correlated_effective_sources"] == contract["correlated_relation_effective_sources"]
    assert abs(h["relation_correlated_confidence"] - contract["correlated_relation_confidence"]) < 1e-12
    assert h["relation_independent_effective_sources"] == contract["independent_relation_effective_sources"]
    assert abs(h["relation_independent_confidence"] - contract["independent_relation_confidence"]) < 1e-12
    assert h["sense_correlated_status"] == contract["sense_correlated_status"]
    assert h["sense_independent_status"] == contract["sense_independent_status"]
    assert h["exception_correlated_status"] == contract["exception_correlated_status"]
    assert h["exception_independent_status"] == contract["exception_independent_status"]
    assert h["all_controls_pass"] is True
    assert all(report.controls.values())

print("FORGE_V38_GLOBAL_LINEAGE_CONSENSUS_OK")
