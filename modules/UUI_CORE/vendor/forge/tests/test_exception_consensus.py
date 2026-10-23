from pathlib import Path
import tempfile, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from equinox_forge.database import CanonicalDB
from equinox_forge.teacher_behavior import TeacherCandidate, BehaviourProbeConfig, extract_teacher_behaviour
from equinox_forge.frame_induction import FrameInductionPolicy, induce_relation_frames, persist_frame_document
from equinox_forge.prototype_hierarchy import PrototypeHierarchyPolicy, induce_prototype_hierarchy
from equinox_forge.hierarchical_frame_transfer import HierarchicalFrameTransferPolicy, promote_frames_to_higher_concepts
from equinox_forge.exception_induction import ExceptionInductionPolicy, induce_exception_overrides
from equinox_forge.exception_consensus import ExceptionConsensusPolicy, track_exception_run, list_exception_states
from equinox_forge.canonical import make_text_surface
from equinox_forge.ir import ForgeDocument, RelationEvidence, RelationType, SourceRef
from equinox_forge.generalization import _pass_through_compression
from equinox_forge.compiler import CompilePolicy
from equinox_forge.runtime_pack import compile_base_pack, load_base_pack
from equinox_forge.runtime_cognition import RuntimeCognition, RuntimeCognitionPolicy

class BaseTeacher:
    backend_id='v029-base'
    def manifest(self): return {'backend_id':self.backend_id}
    def next_token(self,prompt,top_k,temperature):
        table={
            'Hund atmet':[(' Luft',.92),(' Sauerstoff',.08)],
            'Katze atmet':[(' Luft',.92),(' Sauerstoff',.08)],
            'Adler fliegt':[(' Himmel',.91),(' hoch',.09)],
            'Falke fliegt':[(' Himmel',.91),(' hoch',.09)],
        }
        return [TeacherCandidate(i+1,s,s,p,3-i) for i,(s,p) in enumerate(table[prompt.strip()][:top_k])]

class ExceptionTeacher:
    def __init__(self, support: bool, name: str): self.support=support; self.backend_id=name
    def manifest(self): return {'backend_id':self.backend_id}
    def next_token(self,prompt,top_k,temperature):
        vals=[(' Sauerstoff',.86),(' Luft',.14)] if self.support else [(' Luft',.90),(' Sauerstoff',.10)]
        return [TeacherCandidate(100+i,s,s,p,4-i) for i,(s,p) in enumerate(vals[:top_k])]

def semantic_doc():
    names=['Hund','Katze','Adler','Falke','Eule','Stein']; nodes={}; surfaces={}
    for text in names:
        qs,n=make_text_surface(text)
        for q in qs: nodes[q.node_id]=q
        nodes[n.node_id]=n; surfaces[text]=n
    src=SourceRef('v029-semantic','1','independent_semantic_fixture')
    pairs=[('Hund','Katze',.98),('Adler','Falke',.98),('Hund','Adler',.91),('Hund','Falke',.90),('Katze','Adler',.90),('Katze','Falke',.91),('Eule','Adler',.97),('Eule','Falke',.96),('Eule','Hund',.90),('Eule','Katze',.90),('Stein','Hund',.05),('Stein','Katze',.04),('Stein','Adler',.03),('Stein','Falke',.04)]
    ev=[RelationEvidence(surfaces[a].node_id,surfaces[b].node_id,RelationType.SEMANTIC_SIMILARITY.value,score,.99,src,{'independent':True}) for a,b,score in pairs]
    return ForgeDocument(nodes=sorted(nodes.values(),key=lambda n:(n.level,n.node_id)),evidence=ev)

def ingest(db,doc):
    for n in doc.nodes: db.upsert_node(n)
    for e in doc.evidence: db.add_evidence(e)
    db.commit()

def compile_pack(db,path,version):
    comp=_pass_through_compression(db.db)
    manifest=compile_base_pack(db.db,path,pack_id=f'v029-{version}',pack_version=29,compression_run_id=comp.compression_run_id,include_metadata=True,relation_policy=CompilePolicy(min_score=0,min_confidence=0,min_sources=1))
    assert manifest.roundtrip_valid
    return load_base_pack(path)

def predict(pack):
    pol=RuntimeCognitionPolicy(max_recall_nodes=64,max_recall_depth=2,max_predictions=4,min_prediction_score=0.0,
        enable_exact_context=False,enable_suffix_context=False,enable_lexical_seeds=False,enable_parent_routing=False,
        enable_relation_recall=False,enable_transition_prediction=False,enable_recalled_context_prediction=False,
        enable_direct_context_prediction=True,enable_frame_schema_matching=True,enable_semantic_slot_admission=True,
        enable_slot_prototype_admission=True,enable_hierarchical_slot_admission=True,enable_frame_specificity_override=True,
        enable_exception_override_frames=True,enable_exception_lifecycle_gating=True)
    r=RuntimeCognition(pack,pol).predict('Eule atmet')
    return r.predictions[0].surface if r.predictions else None

cfg=BehaviourProbeConfig(top_k=2,min_probability=0.0)
base_doc,_,_=extract_teacher_behaviour(BaseTeacher(),['Hund atmet','Katze atmet','Adler fliegt','Falke fliegt'],'v029-base','1',cfg)
sup_a,_,_=extract_teacher_behaviour(ExceptionTeacher(True,'support-A'),['Adler atmet','Falke atmet'],'support-A','1',cfg)
sup_b,_,_=extract_teacher_behaviour(ExceptionTeacher(True,'support-B'),['Adler atmet','Falke atmet'],'support-B','1',cfg)
con_c,_,_=extract_teacher_behaviour(ExceptionTeacher(False,'conflict-C'),['Adler atmet','Falke atmet'],'conflict-C','1',cfg)
con_d,_,_=extract_teacher_behaviour(ExceptionTeacher(False,'conflict-D'),['Adler atmet','Falke atmet'],'conflict-D','1',cfg)

cons=ExceptionConsensusPolicy(stable_min_sources=2,stable_min_runs=2,stable_consensus=.75,stable_min_support_weight=1.5,quarantine_conflicts=2,quarantine_conflict_weight=1.5)
with tempfile.TemporaryDirectory() as td:
    root=Path(td)
    with CanonicalDB(root/'master.sqlite') as db:
        ingest(db,base_doc); ingest(db,semantic_doc())
        fdoc,_=induce_relation_frames(db.db,FrameInductionPolicy(enable_semantic_slot_typing=True,min_filler_coherence_score=.95,min_filler_pair_fraction=1.0,require_typed_slot=True,enable_latent_slot_prototype=True,prototype_classify_candidates=True,prototype_min_admission_score=.95,prototype_min_support_fillers=2,prototype_membership_confidence=1.0))
        persist_frame_document(db.db,fdoc); db.db.commit()
        hdoc,_=induce_prototype_hierarchy(db.db,PrototypeHierarchyPolicy(max_depth=1,min_cross_leaf_score=.85,score_decay_per_depth=0,min_score_floor=.85,min_cross_leaf_pair_fraction=1.0,membership_confidence=.95))
        persist_frame_document(db.db,hdoc); db.db.commit()
        tdoc,_=promote_frames_to_higher_concepts(db.db,HierarchicalFrameTransferPolicy(min_ancestor_hops=1,max_ancestor_hops=2,min_path_edge_quality=.50,hierarchical_admission_min_path_score=.70))
        persist_frame_document(db.db,tdoc); db.db.commit()
        ingest(db,sup_a); ingest(db,sup_b)
        odoc,orep=induce_exception_overrides(db.db,ExceptionInductionPolicy(min_exception_target_score=.70,min_disagreement_margin=.50,min_membership_quality=.80,min_prototype_path_quality=.50,max_prototype_hops=2))
        assert orep.overrides_induced==1
        persist_frame_document(db.db,odoc); db.db.commit()

        # Two independent supporters are not enough on the first sighting.
        r1=track_exception_run(db.db,odoc,cons)
        assert {x['status'] for x in r1.assessments}=={'candidate'}, r1.to_dict()
        assert predict(compile_pack(db,root/'candidate.eqbase','candidate'))=='Luft'
        # Repeated clean support promotes to stable and enables runtime override.
        r2=track_exception_run(db.db,odoc,cons)
        assert {x['status'] for x in r2.assessments}=={'stable'}, r2.to_dict()
        assert predict(compile_pack(db,root/'stable.eqbase','stable'))=='Sauerstoff'

        ingest(db,con_c)
        r3=track_exception_run(db.db,odoc,cons)
        assert {x['status'] for x in r3.assessments}=={'contested'}, r3.to_dict()
        assert predict(compile_pack(db,root/'contested.eqbase','contested'))=='Luft'

        ingest(db,con_d)
        r4=track_exception_run(db.db,odoc,cons)
        assert {x['status'] for x in r4.assessments}=={'quarantined'}, r4.to_dict()
        assert predict(compile_pack(db,root/'quarantine.eqbase','quarantine'))=='Luft'

        # Disable the two conflicting sources. Recovery itself must again be repeated:
        # first clean rebuild -> candidate, second -> stable.
        for src in ('conflict-C','conflict-D'):
            db.db.execute("INSERT OR REPLACE INTO source_policies(source_id,source_version,enabled,trust_weight,note,metadata_json) VALUES(?,?,?,?,?,?)",(src,'1',0,1.0,'v0.29 recovery fixture','{}'))
        db.db.commit()
        r5=track_exception_run(db.db,odoc,cons)
        assert {x['status'] for x in r5.assessments}=={'candidate'}, r5.to_dict()
        r6=track_exception_run(db.db,odoc,cons)
        assert {x['status'] for x in r6.assessments}=={'stable'}, r6.to_dict()
        assert predict(compile_pack(db,root/'recovered.eqbase','recovered'))=='Sauerstoff'
        states=list_exception_states(db.db)
        assert len(states)==1 and states[0]['status']=='stable'
        stats=db.stats(); assert stats['exception_hypothesis_runs']==6 and stats['exception_hypotheses']==1

print('FORGE_V29_EXCEPTION_CONSENSUS_OK')
