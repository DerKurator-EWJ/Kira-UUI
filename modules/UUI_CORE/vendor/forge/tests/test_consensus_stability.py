from pathlib import Path
import tempfile, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from equinox_forge.hidden_state import HiddenStateSample, HiddenStateConfig, load_probe_dataset, extract_contextual_hidden_evidence
from equinox_forge.database import CanonicalDB
from equinox_forge.ir import RelationEvidence, RelationType, SourceRef
from equinox_forge.sense_compiler import SenseCompilePolicy, FusionWeights, compile_provisional_senses, find_text_surface_id
from equinox_forge.consensus import ConsensusPolicy, track_hypothesis_run, list_hypothesis_states

class StructureBackend:
    backend_id='structure-only-v08'
    def manifest(self): return {'backend_id':self.backend_id,'dimension':4}
    def contextual_state(self,text,target_start,target_end,layers,pooling):
        # Vectors are only used to build canonical target/context nodes. Similarity
        # evidence is removed below so source A/B/C/D fully control consensus.
        return [HiddenStateSample(layer=l,vector=[1.0,0.0,0.0,0.0],token_count=1,metadata={}) for l in layers]

probes=load_probe_dataset([ROOT/'examples'/'probes'/'bank_senses.jsonl'])
doc,_,_=extract_contextual_hidden_evidence(
    StructureBackend(), probes, 'fixture-use','1', HiddenStateConfig(layers=(-1,),top_k=6,min_cosine=-1.0)
)
# Keep canonical contexts + contextual_use, discard generated pair similarity.
doc.evidence=[e for e in doc.evidence if e.relation_type==RelationType.CONTEXTUAL_USE.value]
probe_label={p.probe_id:p.label for p in probes}
ctx_label={}
for e in doc.evidence:
    ctx_label[e.to_id]=probe_label[e.metadata['probe_id']]
finance=sorted([c for c,l in ctx_label.items() if l=='finance'])
seat=sorted([c for c,l in ctx_label.items() if l=='seat'])
assert len(finance)==len(seat)==3


def add_pair_source(db, source_id, internal_score, cross_score=None):
    src=SourceRef(source_id,'1','contextual_hidden_similarity')
    groups=[finance,seat]
    for group in groups:
        for i,a in enumerate(group):
            for b in group[i+1:]:
                db.add_evidence(RelationEvidence(a,b,RelationType.CONTEXTUAL_SIMILARITY.value,internal_score,1.0,src,{'fixture':True}))
    if cross_score is not None:
        for a in finance:
            for b in seat:
                db.add_evidence(RelationEvidence(a,b,RelationType.CONTEXTUAL_SIMILARITY.value,cross_score,1.0,src,{'fixture':True,'conflict':True}))

sense_policy=SenseCompilePolicy(
    threshold=.75,min_channels=1,min_contexts=2,
    weights=FusionWeights(hidden_similarity=1.0,lexical_overlap=0.0,semantic_similarity=0.0,substitution=0.0,transition=0.0,prediction_signature=0.0),
)
cons_policy=ConsensusPolicy(
    source_vote_threshold=.58,source_vote_margin=.08,min_observed_pairs=1,
    stable_min_sources=2,stable_min_runs=2,stable_consensus=.75,
    quarantine_conflicts=2,quarantine_after_misses=2,
)

with tempfile.TemporaryDirectory() as td:
    dbp=Path(td)/'consensus.sqlite'
    with CanonicalDB(dbp) as db:
        for n in doc.nodes: db.upsert_node(n)
        for e in doc.evidence: db.add_evidence(e)
        add_pair_source(db,'teacher-A',.95)
        add_pair_source(db,'teacher-B',.93)
        db.commit()
        target=find_text_surface_id(db.db,'Bank')

        # First sighting: two independent supporters, but no temporal stability yet.
        d1,_=compile_provisional_senses(db.db,target,sense_policy)
        r1=track_hypothesis_run(db.db,d1,target,sense_policy,cons_policy)
        assert len(r1.hypotheses)==2, r1.to_dict()
        assert {h['status'] for h in r1.hypotheses}=={'candidate'}
        assert all(h['support_sources']>=2 and h['conflict_sources']==0 for h in r1.hypotheses)

        # Second identical rebuild promotes both to stable.
        d2,_=compile_provisional_senses(db.db,target,sense_policy)
        r2=track_hypothesis_run(db.db,d2,target,sense_policy,cons_policy)
        assert {h['status'] for h in r2.hypotheses}=={'stable'}, r2.to_dict()
        stable_ids={h['hypothesis_id'] for h in r2.hypotheses}
        assert stable_ids=={n.node_id for n in d2.nodes if n.metadata.get('lifecycle_status')=='stable'}

        # One teacher now prefers cross-sense coupling. Cross score stays below the
        # sense threshold, so the same hypotheses remain but become contested.
        add_pair_source(db,'teacher-C',.58,.70)
        db.commit()
        d3,_=compile_provisional_senses(db.db,target,sense_policy)
        r3=track_hypothesis_run(db.db,d3,target,sense_policy,cons_policy)
        assert {h['hypothesis_id'] for h in r3.hypotheses}==stable_ids
        assert {h['status'] for h in r3.hypotheses}=={'contested'}, r3.to_dict()
        assert all(h['conflict_sources']>=1 for h in r3.hypotheses)

        # A second independent conflicting source causes quarantine.
        add_pair_source(db,'teacher-D',.58,.70)
        db.commit()
        d4,_=compile_provisional_senses(db.db,target,sense_policy)
        r4=track_hypothesis_run(db.db,d4,target,sense_policy,cons_policy)
        assert {h['status'] for h in r4.hypotheses}=={'quarantined'}, r4.to_dict()
        assert all(h['conflict_sources']>=2 for h in r4.hypotheses)

        states=list_hypothesis_states(db.db,target)
        assert len(states)==2
        assert {x['status'] for x in states}=={'quarantined'}
        stats=db.stats()
        assert stats['hypothesis_runs']==4 and stats['hypotheses']==2, stats

print('FORGE_V08_CONSENSUS_STABILITY_OK')
