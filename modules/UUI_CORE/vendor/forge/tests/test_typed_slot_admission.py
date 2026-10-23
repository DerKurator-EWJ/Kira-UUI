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


class ScriptedTypedSlotTeacher:
    backend_id='scripted-typed-slot-teacher'
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
        }
        vals=table[p][:top_k]
        return [TeacherCandidate(i+1,s,s,prob,4.0-i) for i,(s,prob) in enumerate(vals)]


def semantic_support_doc():
    nodes={}
    surfaces={}
    for text in ['Hund','Katze','Wolf','Fuchs','Stein']:
        qs,node=make_text_surface(text)
        for q in qs: nodes[q.node_id]=q
        nodes[node.node_id]=node
        surfaces[text]=node
    src=SourceRef('independent-semantic-graph','1','semantic_similarity_fixture')
    pairs=[
        ('Hund','Katze',.92),
        ('Wolf','Hund',.94),('Wolf','Katze',.90),
        ('Fuchs','Hund',.91),('Fuchs','Katze',.89),
        # Stein is deliberately not semantically supported by either slot filler.
        ('Stein','Hund',.08),('Stein','Katze',.06),
    ]
    ev=[]
    for a,b,score in pairs:
        ev.append(RelationEvidence(surfaces[a].node_id,surfaces[b].node_id,RelationType.SEMANTIC_SIMILARITY.value,score,.98,src,
                                   {'independent_of_frame_teacher':True}))
    return ForgeDocument(nodes=sorted(nodes.values(), key=lambda n:(n.level,n.node_id)), evidence=ev)


with tempfile.TemporaryDirectory() as td:
    out=Path(td)/'typed-slot'
    cfg=BehaviourProbeConfig(top_k=3,min_probability=0.0,evidence_confidence=0.95)
    runtime=RuntimeCognitionPolicy(
        max_recall_nodes=96,max_recall_depth=3,max_predictions=3,min_prediction_score=0.0,
        enable_semantic_slot_admission=True,
    )
    frame_policy=FrameInductionPolicy(
        min_target_context_fraction=1.0,
        enable_semantic_slot_typing=True,
        min_filler_coherence_score=0.70,
        min_filler_pair_fraction=1.0,
        slot_admission_min_score=0.70,
        slot_admission_min_support_fillers=2,
        require_typed_slot=True,
    )
    report=compile_and_benchmark_frame_schemas(
        ScriptedTypedSlotTeacher(),
        ['Hund ist ein','Katze ist ein'],
        [
            FrameEvalCase('Wolf ist ein',True,'known lexical concept; unseen predictive context'),
            FrameEvalCase('Fuchs ist ein',True,'second semantically admitted filler'),
            FrameEvalCase('Mein Wolf ist ein',True,'new outer prefix plus typed frame suffix'),
            FrameEvalCase('Stein ist ein',False,'structural match but semantically inadmissible filler'),
        ],
        out,'scripted-typed-slot','1',probe_cfg=cfg,frame_policy=frame_policy,cognition_policy=runtime,
        runtime_top_k=3,support_documents=[semantic_support_doc()],pack_version=23,
    )
    assert report.induced_frames == 1
    assert report.semantic_admission_enabled is True
    assert report.support_documents == 1
    assert report.frame_applicability_recall == 1.0
    assert report.frame_applicability_precision == 1.0
    # Same BASE with the semantic gate disabled reproduces v0.22's open-slot false positive.
    assert abs(report.open_frame_baseline['top1_retention'] - 0.75) < 1e-12
    assert abs(report.frame_only['top1_retention'] - 0.75) < 1e-12

    detail={x.prompt:x for x in report.details}
    for prompt in ['Wolf ist ein','Fuchs ist ein','Mein Wolf ist ein']:
        assert detail[prompt].frame_matched is True
        assert detail[prompt].frame_only_top1 == 'Tier'
        assert detail[prompt].frame_only_top1_match is True
    stone=detail['Stein ist ein']
    assert stone.open_frame_top1 == 'Tier'          # old untyped behavior
    assert stone.frame_only_top1 is None            # v0.23 abstains instead of hallucinating Tier
    assert stone.frame_matched is False
    assert stone.teacher_top1 == 'Objekt'
    reasons=[sc.get('reason') for row in stone.frame_admission for sc in row.get('slot_checks',[])]
    assert 'insufficient_semantic_support' in reasons, reasons

    pack=load_base_pack(out/'train-with-frames.eqbase')
    cog=RuntimeCognition(pack,runtime)
    # The concepts are independently known, but the held-out predictive contexts are not compiled.
    assert 'Wolf' in cog.index.surface_to_nodes
    assert 'Fuchs' in cog.index.surface_to_nodes
    assert 'Stein' in cog.index.surface_to_nodes
    assert 'Wolf ist ein' not in cog.index.surface_to_nodes
    assert 'Fuchs ist ein' not in cog.index.surface_to_nodes
    assert 'Stein ist ein' not in cog.index.surface_to_nodes

    frame_nodes=[n for n in pack.nodes if (n.metadata or {}).get('role')=='induced_frame_schema']
    assert len(frame_nodes)==1
    slot_ids=(frame_nodes[0].metadata or {}).get('slot_node_ids',[])
    assert len(slot_ids)==1
    slot=cog.index.node_by_id[slot_ids[0]]
    profile=(slot.metadata or {}).get('semantic_admission')
    assert profile and profile['enabled'] is True
    assert profile['min_support_fillers']==2
    assert set(profile['profile_filler_surfaces'])=={'Hund','Katze'}

print('FORGE_V23_TYPED_SLOT_ADMISSION_OK')
