from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from equinox_forge.hidden_state import (
    HiddenStateSample, HiddenStateConfig, load_probe_dataset,
    extract_contextual_hidden_evidence,
)
from equinox_forge.validate import validate_document
from equinox_forge.database import CanonicalDB
from equinox_forge.ir import RelationType

class ScriptedHiddenBackend:
    backend_id='scripted-contextual-hidden'
    def manifest(self): return {'backend_id':self.backend_id,'dimension':4}
    def contextual_state(self,text,target_start,target_end,layers,pooling):
        lo=text.lower()
        if any(k in lo for k in ['geld','kredit','konto','überweist']):
            base=[1.0,0.05,0.0,0.0]
        else:
            base=[0.05,1.0,0.0,0.0]
        return [HiddenStateSample(layer=l,vector=base,token_count=1,metadata={'fixture':True}) for l in layers]

probes=load_probe_dataset([ROOT/'examples'/'probes'/'bank_senses.jsonl'])
assert len(probes)==6
cfg=HiddenStateConfig(layers=(-1,),top_k=3,min_cosine=0.20)
doc,report,records=extract_contextual_hidden_evidence(ScriptedHiddenBackend(),probes,'fixture-hidden','1',cfg)
errors=[x for x in validate_document(doc) if x.severity=='error']
assert not errors, errors
assert report.probes_encoded==6
assert report.surfaces==1
assert report.context_nodes==6
assert report.contextual_use_evidence==6
assert report.contextual_similarity_evidence==6, report.to_dict()  # 3 choose 2 per sense
assert report.mean_cosine_same_label is not None and report.mean_cosine_same_label > .99
assert report.mean_cosine_different_label is not None and report.mean_cosine_different_label < .2
assert report.nearest_neighbour_label_accuracy==1.0
assert all(r.vector is None for r in records)
assert any(e.relation_type==RelationType.CONTEXTUAL_USE.value for e in doc.evidence)
assert any(e.relation_type==RelationType.CONTEXTUAL_SIMILARITY.value for e in doc.evidence)

# labels are evaluation-only; removing labels must not change canonical node IDs/evidence endpoints.
unlabelled=[type(p)(p.probe_id,p.text,p.target,p.occurrence,None,p.metadata) for p in probes]
doc2,report2,_=extract_contextual_hidden_evidence(ScriptedHiddenBackend(),unlabelled,'fixture-hidden','1',cfg)
assert {n.node_id for n in doc.nodes} == {n.node_id for n in doc2.nodes}
assert {(e.from_id,e.to_id,e.relation_type,round(e.score,8),str(sorted(e.metadata.items()))) for e in doc.evidence} == {(e.from_id,e.to_id,e.relation_type,round(e.score,8),str(sorted(e.metadata.items()))) for e in doc2.evidence}

with CanonicalDB(ROOT/'out'/'test_hidden.sqlite') as db:
    for n in doc.nodes: db.upsert_node(n)
    for e in doc.evidence: db.add_evidence(e)
    db.commit(); first=db.stats()['evidence']
    for e in doc.evidence: db.add_evidence(e)
    db.commit(); second=db.stats()['evidence']
    assert first==second
print('FORGE_V06_HIDDEN_EVIDENCE_OK')
