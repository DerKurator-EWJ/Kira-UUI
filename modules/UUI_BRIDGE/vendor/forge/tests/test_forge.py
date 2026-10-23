from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tempfile
from equinox_forge.canonical import make_text_surface, tier_name
from equinox_forge.demo_builder import build_demo
from equinox_forge.validate import validate_document
from equinox_forge.database import CanonicalDB
from equinox_forge.compiler import aggregate_relations

def run():
    q1,hund1=make_text_surface("Hund")
    q2,hund2=make_text_surface("Hund")
    _,hunde=make_text_surface("Hunde")
    assert hund1.node_id==hund2.node_id
    assert hund1.node_id!=hunde.node_id
    assert tier_name(0)=="Q" and tier_name(1)=="F" and tier_name(10)=="Ω" and tier_name(12)=="Ω+2"
    doc=build_demo(); issues=validate_document(doc)
    assert not [x for x in issues if x.severity=="error"], issues
    with tempfile.TemporaryDirectory() as td:
        with CanonicalDB(Path(td)/"forge.sqlite") as db:
            for n in sorted(doc.nodes,key=lambda n:(n.level,n.node_id)): db.upsert_node(n)
            # ingest same canonical nodes again must not duplicate nodes
            for n in sorted(doc.nodes,key=lambda n:(n.level,n.node_id)): db.upsert_node(n)
            for e in doc.evidence: db.add_evidence(e)
            db.commit()
            stats=db.stats(); assert stats["nodes"]==len({n.node_id for n in doc.nodes})
            rel=aggregate_relations(db.db)
            isa=[r for r in rel if r.relation_type=="is_a"]
            assert len(isa)==1 and isa[0].source_count==2 and isa[0].evidence_count==2
            assert 0.87 <= isa[0].score <= 0.91
    print(f"FORGE_V01_OK nodes={len(doc.nodes)} evidence={len(doc.evidence)} compiled_relations=2")
if __name__=="__main__": run()
