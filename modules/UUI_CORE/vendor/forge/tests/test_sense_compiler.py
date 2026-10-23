from pathlib import Path
import sys, tempfile, json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from equinox_forge.hidden_state import HiddenStateSample, HiddenStateConfig, load_probe_dataset, extract_contextual_hidden_evidence
from equinox_forge.database import CanonicalDB
from equinox_forge.ir import RelationEvidence, RelationType, SourceRef, NodeKind
from equinox_forge.sense_compiler import SenseCompilePolicy, FusionWeights, compile_provisional_senses, find_text_surface_id
from equinox_forge.validate import validate_document

class ScriptedHiddenBackend:
    backend_id='scripted-contextual-hidden-v07'
    def manifest(self): return {'backend_id':self.backend_id,'dimension':4}
    def contextual_state(self,text,target_start,target_end,layers,pooling):
        lo=text.lower()
        if any(k in lo for k in ['geld','kredit','konto','überweist']):
            base=[1.0,0.04,0.0,0.0]
        else:
            base=[0.04,1.0,0.0,0.0]
        return [HiddenStateSample(layer=l,vector=base,token_count=1,metadata={'fixture':True}) for l in layers]

probes=load_probe_dataset([ROOT/'examples'/'probes'/'bank_senses.jsonl'])
doc,_,_=extract_contextual_hidden_evidence(
    ScriptedHiddenBackend(),probes,'fixture-hidden','1',
    HiddenStateConfig(layers=(-1,),top_k=3,min_cosine=.20),
)

# Contextual labels are used only here as benchmark truth, never as compiler input.
probe_label={p.probe_id:p.label for p in probes}
ctx_label={}
for e in doc.evidence:
    if e.relation_type==RelationType.CONTEXTUAL_USE.value:
        ctx_label[e.to_id]=probe_label[e.metadata['probe_id']]

node_by_surface={n.surface:n for n in doc.nodes if n.kind==NodeKind.SURFACE.value and n.surface}
sem_pairs=[('Geld','Kredit'),('Kredit','Konto'),('Geld','Konto'),('Park','Baum'),('Baum','See'),('Park','See')]
source=SourceRef('fixture-static','1','embedding_neighbours')

with tempfile.TemporaryDirectory() as td:
    dbp=Path(td)/'sense.sqlite'
    with CanonicalDB(dbp) as db:
        for n in doc.nodes: db.upsert_node(n)
        for e in doc.evidence: db.add_evidence(e)
        for a,b in sem_pairs:
            assert a in node_by_surface and b in node_by_surface, (a,b,node_by_surface.keys())
            for x,y in [(a,b),(b,a)]:
                db.add_evidence(RelationEvidence(
                    node_by_surface[x].node_id,node_by_surface[y].node_id,
                    RelationType.SEMANTIC_SIMILARITY.value,.90,.90,source,
                    {'fixture':True,'channel':'embedding'},
                ))
        db.commit()
        target=find_text_surface_id(db.db,'Bank')
        policy=SenseCompilePolicy(
            threshold=.62,min_channels=2,min_contexts=2,
            weights=FusionWeights(
                hidden_similarity=.65, lexical_overlap=.05,
                semantic_similarity=.25, substitution=.0,
                transition=.0, prediction_signature=.05,
            ),
        )
        derived,report=compile_provisional_senses(db.db,target,policy)
        errors=[x for x in validate_document(derived) if x.severity=='error']
        assert not errors, errors
        assert report.contexts_found==6, report.to_dict()
        assert report.clusters_created==2, report.to_dict()
        assert report.unresolved_contexts==0, report.to_dict()
        assert all(len(c['contexts'])==3 for c in report.clusters)
        cluster_labels=[]
        for c in report.clusters:
            labels={ctx_label[x] for x in c['contexts']}
            cluster_labels.append(labels)
        assert {frozenset(x) for x in cluster_labels}=={frozenset({'finance'}),frozenset({'seat'})},cluster_labels
        assert all(n.kind==NodeKind.CONCEPT.value for n in derived.nodes if n.metadata.get('provisional'))
        # Gold labels must never leak into derived IR/report compiler inputs.
        payload=json.dumps(derived.to_dict(),ensure_ascii=False).lower()
        assert 'finance' not in payload and 'seat' not in payload
        first_ids={n.node_id for n in derived.nodes if n.metadata.get('provisional')}
        derived2,report2=compile_provisional_senses(db.db,target,policy)
        second_ids={n.node_id for n in derived2.nodes if n.metadata.get('provisional')}
        assert first_ids==second_ids
        # Derived results can be ingested idempotently.
        for n in derived.nodes: db.upsert_node(n)
        for e in derived.evidence: db.add_evidence(e)
        db.commit(); first=db.stats()['evidence']
        for n in derived.nodes: db.upsert_node(n)
        for e in derived.evidence: db.add_evidence(e)
        db.commit(); second=db.stats()['evidence']
        assert first==second

print('FORGE_V07_SENSE_FUSION_OK')
