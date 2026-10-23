from pathlib import Path
import tempfile, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from equinox_forge.teacher_behavior import TeacherCandidate, BehaviourProbeConfig
from equinox_forge.runtime_cognition import RuntimeCognitionPolicy, RuntimeCognition
from equinox_forge.runtime_pack import load_base_pack
from equinox_forge.frame_induction import FrameInductionPolicy
from equinox_forge.frame_schema_benchmark import FrameEvalCase, compile_and_benchmark_frame_schemas
from equinox_forge.canonical import make_text_surface
from equinox_forge.ir import ForgeDocument, RelationEvidence, RelationType, SourceRef


class ScriptedPrototypeTeacher:
    backend_id='scripted-prototype-teacher'
    def manifest(self): return {'backend_id':self.backend_id,'fixture':True}
    def next_token(self,prompt,top_k,temperature):
        p=prompt.strip()
        table={
            'Hund ist ein':[(' Tier',.82),(' Haustier',.12),(' Wesen',.06)],
            'Katze ist ein':[(' Tier',.82),(' Haustier',.12),(' Wesen',.06)],
            'Wolf ist ein':[(' Tier',.84),(' Wesen',.10),(' Raubtier',.06)],
            'Fuchs ist ein':[(' Tier',.84),(' Wesen',.10),(' Raubtier',.06)],
            'Mein Wolf ist ein':[(' Tier',.84),(' Wesen',.10),(' Raubtier',.06)],
            'Stein ist ein':[(' Objekt',.80),(' Material',.12),(' Ding',.08)],
            'Hund braucht':[(' Futter',.86),(' Nahrung',.09),(' Wasser',.05)],
            'Katze braucht':[(' Futter',.86),(' Nahrung',.09),(' Wasser',.05)],
        }
        vals=table[p][:top_k]
        return [TeacherCandidate(i+1,s,s,prob,4.0-i) for i,(s,prob) in enumerate(vals)]


def semantic_support_doc():
    nodes={}; surfaces={}
    for text in ['Hund','Katze','Wolf','Fuchs','Stein']:
        qs,node=make_text_surface(text)
        for q in qs: nodes[q.node_id]=q
        nodes[node.node_id]=node; surfaces[text]=node
    src=SourceRef('independent-semantic-graph','1','semantic_similarity_fixture')
    pairs=[
        ('Hund','Katze',.92),
        ('Wolf','Hund',.94),('Wolf','Katze',.90),
        ('Fuchs','Hund',.91),('Fuchs','Katze',.89),
        ('Stein','Hund',.08),('Stein','Katze',.06),
    ]
    ev=[RelationEvidence(surfaces[a].node_id,surfaces[b].node_id,RelationType.SEMANTIC_SIMILARITY.value,score,.98,src,
                         {'independent_of_frame_teacher':True}) for a,b,score in pairs]
    return ForgeDocument(nodes=sorted(nodes.values(), key=lambda n:(n.level,n.node_id)), evidence=ev)


with tempfile.TemporaryDirectory() as td:
    out=Path(td)/'slot-prototype'
    cfg=BehaviourProbeConfig(top_k=3,min_probability=0.0,evidence_confidence=0.95)
    runtime=RuntimeCognitionPolicy(
        max_recall_nodes=96,max_recall_depth=3,max_predictions=3,min_prediction_score=0.0,
        enable_semantic_slot_admission=True,enable_slot_prototype_admission=True,
    )
    frame_policy=FrameInductionPolicy(
        min_target_context_fraction=1.0,
        enable_semantic_slot_typing=True,
        min_filler_coherence_score=0.70,
        min_filler_pair_fraction=1.0,
        slot_admission_min_score=0.70,
        slot_admission_min_support_fillers=2,
        require_typed_slot=True,
        enable_latent_slot_prototype=True,
        prototype_classify_candidates=True,
        prototype_min_admission_score=0.70,
        prototype_min_support_fillers=2,
        prototype_membership_confidence=0.95,
    )
    report=compile_and_benchmark_frame_schemas(
        ScriptedPrototypeTeacher(),
        ['Hund ist ein','Katze ist ein','Hund braucht','Katze braucht'],
        [
            FrameEvalCase('Wolf ist ein',True,'known concept; unseen predictive context'),
            FrameEvalCase('Fuchs ist ein',True,'second admitted prototype member'),
            FrameEvalCase('Mein Wolf ist ein',True,'prefix transfer through the same prototype-typed frame'),
            FrameEvalCase('Stein ist ein',False,'structural frame match but no latent-class membership'),
        ],
        out,'scripted-prototype','1',probe_cfg=cfg,frame_policy=frame_policy,cognition_policy=runtime,
        runtime_top_k=3,support_documents=[semantic_support_doc()],pack_version=24,
    )
    assert report.induced_frames == 2
    assert report.frame_induction['generated_slot_prototypes'] == 1  # reused by both frames
    assert report.frame_induction['generated_prototype_memberships'] == 4  # Hund, Katze, Wolf, Fuchs
    assert report.frame_applicability_recall == 1.0
    assert report.frame_applicability_precision == 1.0

    pack=load_base_pack(out/'train-with-frames.eqbase')
    cog=RuntimeCognition(pack,runtime)
    prototypes=[n for n in pack.nodes if (n.metadata or {}).get('role')=='latent_slot_prototype']
    assert len(prototypes)==1
    frames=[n for n in pack.nodes if (n.metadata or {}).get('role')=='induced_frame_schema']
    assert len(frames)==2
    proto=prototypes[0]
    for frame in frames:
        slot_ids=(frame.metadata or {}).get('slot_node_ids',[])
        assert len(slot_ids)==1
        slot=cog.index.node_by_id[slot_ids[0]]
        assert (slot.metadata or {}).get('latent_prototype_node_id')==proto.node_id
    assert set((proto.metadata or {}).get('filler_surfaces',[]))=={'Hund','Katze'}

    def surface_node(surface):
        return next(nid for nid in cog.index.surface_to_nodes[surface] if cog.index.node_by_id[nid].kind=='surface')
    wolf=surface_node('Wolf'); fuchs=surface_node('Fuchs'); stein=surface_node('Stein')
    def memberships(nid):
        return [r for r in cog.index.relations_from.get(nid,[]) if r.relation_type==RelationType.IS_A.value and r.to_id==proto.node_id]
    assert len(memberships(wolf))==1
    assert len(memberships(fuchs))==1
    assert memberships(stein)==[]

    wolf_pred=cog.predict('Wolf ist ein')
    checks=[sc for row in wolf_pred.frame_admission for sc in row.get('slot_checks',[])]
    assert any(sc.get('reason')=='prototype_membership' and sc.get('prototype_node_id')==proto.node_id and sc.get('prototype_index_lookups')==1 for sc in checks), checks
    stone_pred=cog.predict('Stein ist ein')
    checks=[sc for row in stone_pred.frame_admission for sc in row.get('slot_checks',[])]
    assert any(sc.get('reason')=='insufficient_prototype_membership' for sc in checks), checks

    # The prototype path must work without pairwise semantic similarity at query time:
    # only the compiled candidate -> Concept_X membership edge is needed for admission.
    pairwise_off=RuntimeCognition(pack,RuntimeCognitionPolicy(
        max_recall_nodes=96,max_recall_depth=3,max_predictions=3,min_prediction_score=0.0,
        enable_semantic_slot_admission=True,enable_slot_prototype_admission=True,
        relation_weights={},
    ))
    assert pairwise_off.predict('Wolf ist ein').predictions[0].surface=='Tier'

print('FORGE_V24_SLOT_PROTOTYPE_OK')
