from pathlib import Path
import tempfile, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from equinox_forge.database import CanonicalDB
from equinox_forge.teacher_behavior import TeacherCandidate, BehaviourProbeConfig, extract_teacher_behaviour
from equinox_forge.frame_induction import FrameInductionPolicy, induce_relation_frames, persist_frame_document
from equinox_forge.prototype_hierarchy import PrototypeHierarchyPolicy, induce_prototype_hierarchy
from equinox_forge.hierarchical_frame_transfer import HierarchicalFrameTransferPolicy, promote_frames_to_higher_concepts
from equinox_forge.canonical import make_text_surface, composite_id
from equinox_forge.ir import ForgeDocument, IRNode, Modality, NodeKind, RelationEvidence, RelationType, SourceRef
from equinox_forge.generalization import _pass_through_compression
from equinox_forge.compiler import CompilePolicy
from equinox_forge.runtime_pack import compile_base_pack, load_base_pack
from equinox_forge.runtime_cognition import RuntimeCognition, RuntimeCognitionPolicy
from equinox_forge.specificity_resolution import SpecificityCase, benchmark_specificity_resolution


class ScriptedTeacher:
    backend_id = 'scripted-specificity-teacher'
    def manifest(self): return {'backend_id': self.backend_id, 'fixture': True}
    def next_token(self, prompt, top_k, temperature):
        table = {
            'Hund atmet': [(' Luft', .92), (' Sauerstoff', .08)],
            'Katze atmet': [(' Luft', .92), (' Sauerstoff', .08)],
            'Adler fliegt': [(' Himmel', .91), (' hoch', .09)],
            'Falke fliegt': [(' Himmel', .91), (' hoch', .09)],
        }
        vals = table[prompt.strip()][:top_k]
        return [TeacherCandidate(i + 1, s, s, prob, 3.0 - i) for i, (s, prob) in enumerate(vals)]


def semantic_support_doc():
    names = ['Hund','Katze','Adler','Falke','Stein']
    nodes = {}; surfaces = {}
    for text in names:
        qs, node = make_text_surface(text)
        for q in qs: nodes[q.node_id] = q
        nodes[node.node_id] = node
        surfaces[text] = node
    src = SourceRef('independent-semantic-specificity', '1', 'semantic_similarity_fixture')
    pairs = [
        ('Hund','Katze',.98), ('Adler','Falke',.98),
        ('Hund','Adler',.91),('Hund','Falke',.90),('Katze','Adler',.90),('Katze','Falke',.91),
        ('Stein','Hund',.05),('Stein','Katze',.04),('Stein','Adler',.03),('Stein','Falke',.04),
    ]
    ev = [RelationEvidence(surfaces[a].node_id, surfaces[b].node_id, RelationType.SEMANTIC_SIMILARITY.value,
                           score, .99, src, {'independent_of_teacher': True}) for a,b,score in pairs]
    return ForgeDocument(nodes=sorted(nodes.values(), key=lambda n:(n.level,n.node_id)), evidence=ev)


with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    db_path = root/'master.sqlite'
    teacher_doc, _, _ = extract_teacher_behaviour(
        ScriptedTeacher(), ['Hund atmet','Katze atmet','Adler fliegt','Falke fliegt'],
        'teacher-specificity', '1', BehaviourProbeConfig(top_k=2, min_probability=0.0),
    )
    with CanonicalDB(db_path) as db:
        for doc in (teacher_doc, semantic_support_doc()):
            for n in doc.nodes: db.upsert_node(n)
            for e in doc.evidence: db.add_evidence(e)
        db.commit()

        frame_doc, frame_report = induce_relation_frames(db.db, FrameInductionPolicy(
            enable_semantic_slot_typing=True, min_filler_coherence_score=.95,
            min_filler_pair_fraction=1.0, require_typed_slot=True,
            enable_latent_slot_prototype=True, prototype_classify_candidates=False,
            prototype_min_admission_score=.95, prototype_min_support_fillers=2,
        ))
        assert frame_report.generated_slot_prototypes == 2
        # Find the bird-specific slot learned from Adler/Falke in the independent
        # "* fliegt" frame. We will reuse only its slot type to construct a lower-
        # scope exception for the same structural pattern as the broad frame.
        bird_frame = next(n for n in frame_doc.nodes if n.kind == NodeKind.TEMPLATE.value and (n.metadata or {}).get('pattern_tokens') == ['*','fliegt'])
        bird_slot_id = bird_frame.metadata['slot_node_ids'][0]
        bird_slot = next(n for n in frame_doc.nodes if n.node_id == bird_slot_id)
        persist_frame_document(db.db, frame_doc); db.db.commit()

        hierarchy_doc, hierarchy_report = induce_prototype_hierarchy(db.db, PrototypeHierarchyPolicy(
            max_depth=1, min_cross_leaf_score=.85, score_decay_per_depth=0.0,
            min_score_floor=.85, min_cross_leaf_pair_fraction=1.0, membership_confidence=.95,
        ))
        assert hierarchy_report.generated_concepts == 1
        persist_frame_document(db.db, hierarchy_doc); db.db.commit()

        transfer_doc, transfer_report = promote_frames_to_higher_concepts(db.db, HierarchicalFrameTransferPolicy(
            min_ancestor_hops=1, max_ancestor_hops=2, min_path_edge_quality=.50,
            hierarchical_admission_min_path_score=.70,
        ))
        assert transfer_report.frames_promoted == 2
        persist_frame_document(db.db, transfer_doc); db.db.commit()

        # Add a deliberately lower-confidence bird-specific override for the same
        # "* atmet" structure. The higher transferred frame predicts Luft at ~.92;
        # this lower-scope exception predicts Himmel at .60. Without specificity
        # resolution the broad rule wins on score. With v0.27 the lower concept wins
        # categorically, proving override semantics rather than a score bonus.
        atmet_id = db.db.execute("SELECT node_id FROM nodes WHERE kind=? AND surface='atmet' ORDER BY node_id LIMIT 1", (NodeKind.SURFACE.value,)).fetchone()[0]
        himmel_id = db.db.execute("SELECT node_id FROM nodes WHERE kind=? AND surface='Himmel' ORDER BY node_id LIMIT 1", (NodeKind.SURFACE.value,)).fetchone()[0]
        special_id = composite_id(Modality.TEXT.value, NodeKind.TEMPLATE.value, [bird_slot_id, atmet_id], label='specificity:bird-atmet')
        specific_frame = IRNode(
            node_id=special_id, modality=Modality.TEXT.value, kind=NodeKind.TEMPLATE.value,
            level=max(bird_slot.level, 1) + 2, surface='[SLOT_0] atmet', members=[bird_slot_id, atmet_id],
            metadata={
                'role':'induced_frame_schema','provisional':True,'pattern_tokens':['*','atmet'],
                'slot_positions':[0],'slot_node_ids':[bird_slot_id],'open_text_slots':True,
                'support_count':2,'specificity_fixture':'bird_exception','forge_version':'0.27',
            },
        )
        db.upsert_node(specific_frame)
        db.add_evidence(RelationEvidence(
            special_id, himmel_id, RelationType.PREDICTION.value, .60, .99,
            SourceRef('derived:specificity-fixture','0.27','specific_frame_prediction'),
            {'lower_scope_override':True},
        ))
        db.commit()

        comp = _pass_through_compression(db.db)
        manifest = compile_base_pack(
            db.db, root/'specificity.eqbase', pack_id='v027-specificity', pack_version=27,
            compression_run_id=comp.compression_run_id, include_metadata=True,
            relation_policy=CompilePolicy(min_score=0.0, min_confidence=0.0, min_sources=1),
        )
        assert manifest.roundtrip_valid

    pack = load_base_pack(root/'specificity.eqbase')
    isolated = dict(
        max_recall_nodes=64,max_recall_depth=2,max_predictions=4,min_prediction_score=0.0,
        enable_exact_context=False,enable_suffix_context=False,enable_lexical_seeds=False,
        enable_parent_routing=False,enable_relation_recall=False,enable_transition_prediction=False,
        enable_recalled_context_prediction=False,enable_direct_context_prediction=True,
        enable_frame_schema_matching=True,enable_semantic_slot_admission=True,
        enable_slot_prototype_admission=True,enable_hierarchical_slot_admission=True,
    )
    legacy = RuntimeCognition(pack, RuntimeCognitionPolicy(**isolated, enable_frame_specificity_override=False))
    specific = RuntimeCognition(pack, RuntimeCognitionPolicy(**isolated, enable_frame_specificity_override=True))

    # Broad higher concept wins by score in v0.26-style aggregation.
    old = legacy.predict('Adler atmet')
    assert old.predictions and old.predictions[0].surface == 'Luft', [x.to_dict() for x in old.predictions]
    assert len(old.resolved_context_ids) >= 2

    # v0.27: direct bird subtype overrides broad root concept even though its raw
    # prediction score is deliberately lower.
    new = specific.predict('Adler atmet')
    assert new.predictions and new.predictions[0].surface == 'Himmel', [x.to_dict() for x in new.predictions]
    assert len(new.resolved_context_ids) == 1
    assert new.work.get('frame_specificity_suppressed', 0) >= 1
    admitted = [x for x in new.frame_admission if x.get('admitted')]
    assert any(x.get('specificity_scope_class') == 'direct_prototype' and not x.get('suppressed_by_specificity') for x in admitted)
    assert any(x.get('specificity_scope_class') == 'hierarchical_concept' and x.get('suppressed_by_specificity') for x in admitted)

    # A filler with no lower bird membership still gets the broad inherited rule.
    # Hund belongs to the source subtype of that broad frame, so the original direct
    # frame is itself more specific and remains the winner.
    hund = specific.predict('Hund atmet')
    assert hund.predictions and hund.predictions[0].surface == 'Luft'

    # The helper benchmark must compare the exact same pack without recompilation.
    report = benchmark_specificity_resolution(
        pack, [SpecificityCase('Adler atmet','Himmel'), SpecificityCase('Hund atmet','Luft')],
        policy=RuntimeCognitionPolicy(**isolated),
    )
    assert report.cases == 2
    assert report.baseline_top1_accuracy == .5, report.to_dict()
    assert report.specificity_top1_accuracy == 1.0, report.to_dict()
    assert report.improved_cases == 1 and report.regressed_cases == 0
    assert report.specificity_suppressed_frames >= 2

print('FORGE_V27_SPECIFICITY_RESOLUTION_OK')
