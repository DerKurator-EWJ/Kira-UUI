from pathlib import Path
import tempfile

from equinox_forge.database import CanonicalDB
from equinox_forge.teacher_behavior import TeacherCandidate, BehaviourProbeConfig, extract_teacher_behaviour
from equinox_forge.frame_induction import FrameInductionPolicy, induce_relation_frames, persist_frame_document
from equinox_forge.prototype_hierarchy import PrototypeHierarchyPolicy, induce_prototype_hierarchy
from equinox_forge.hierarchical_frame_transfer import HierarchicalFrameTransferPolicy, promote_frames_to_higher_concepts
from equinox_forge.exception_induction import ExceptionInductionPolicy, induce_exception_overrides
from equinox_forge.exception_benchmark import ExceptionCase, benchmark_exception_overrides
from equinox_forge.canonical import make_text_surface
from equinox_forge.ir import ForgeDocument, RelationEvidence, RelationType, SourceRef
from equinox_forge.generalization import _pass_through_compression
from equinox_forge.compiler import CompilePolicy
from equinox_forge.runtime_pack import compile_base_pack, load_base_pack
from equinox_forge.runtime_cognition import RuntimeCognition, RuntimeCognitionPolicy
from equinox_forge.validate import validate_document


class BaselineTeacher:
    backend_id = 'scripted-v028-baseline'
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


class ExceptionTeacher:
    backend_id = 'scripted-v028-exception-observations'
    def manifest(self): return {'backend_id': self.backend_id, 'fixture': True}
    def next_token(self, prompt, top_k, temperature):
        table = {
            'Adler atmet': [(' Sauerstoff', .86), (' Luft', .14)],
            'Falke atmet': [(' Sauerstoff', .84), (' Luft', .16)],
        }
        vals = table[prompt.strip()][:top_k]
        return [TeacherCandidate(100 + i, s, s, prob, 4.0 - i) for i, (s, prob) in enumerate(vals)]


def semantic_support_doc():
    names = ['Hund','Katze','Adler','Falke','Eule','Stein']
    nodes = {}; surfaces = {}
    for text in names:
        qs, node = make_text_surface(text)
        for q in qs: nodes[q.node_id] = q
        nodes[node.node_id] = node
        surfaces[text] = node
    src = SourceRef('independent-semantic-v028', '1', 'semantic_similarity_fixture')
    pairs = [
        ('Hund','Katze',.98), ('Adler','Falke',.98),
        ('Hund','Adler',.91), ('Hund','Falke',.90), ('Katze','Adler',.90), ('Katze','Falke',.91),
        ('Eule','Adler',.97), ('Eule','Falke',.96), ('Eule','Hund',.90), ('Eule','Katze',.90),
        ('Stein','Hund',.05), ('Stein','Katze',.04), ('Stein','Adler',.03), ('Stein','Falke',.04),
    ]
    ev = [RelationEvidence(
        surfaces[a].node_id, surfaces[b].node_id, RelationType.SEMANTIC_SIMILARITY.value,
        score, .99, src, {'independent_of_teacher': True},
    ) for a,b,score in pairs]
    return ForgeDocument(nodes=sorted(nodes.values(), key=lambda n:(n.level,n.node_id)), evidence=ev)


with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    db_path = root/'master.sqlite'
    cfg = BehaviourProbeConfig(top_k=2, min_probability=0.0)
    base_doc, _, _ = extract_teacher_behaviour(
        BaselineTeacher(), ['Hund atmet','Katze atmet','Adler fliegt','Falke fliegt'],
        'teacher-v028-base', '1', cfg,
    )
    exc_doc, _, _ = extract_teacher_behaviour(
        ExceptionTeacher(), ['Adler atmet','Falke atmet'],
        'teacher-v028-exception', '1', cfg,
    )

    with CanonicalDB(db_path) as db:
        for doc in (base_doc, semantic_support_doc()):
            for n in doc.nodes: db.upsert_node(n)
            for e in doc.evidence: db.add_evidence(e)
        db.commit()

        frame_doc, frame_report = induce_relation_frames(db.db, FrameInductionPolicy(
            enable_semantic_slot_typing=True,
            min_filler_coherence_score=.95,
            min_filler_pair_fraction=1.0,
            require_typed_slot=True,
            enable_latent_slot_prototype=True,
            prototype_classify_candidates=True,
            prototype_min_admission_score=.95,
            prototype_min_support_fillers=2,
            prototype_membership_confidence=1.0,
        ))
        assert frame_report.accepted_frames == 2, frame_report.to_dict()
        assert frame_report.generated_slot_prototypes == 2
        persist_frame_document(db.db, frame_doc); db.db.commit()

        hierarchy_doc, hierarchy_report = induce_prototype_hierarchy(db.db, PrototypeHierarchyPolicy(
            max_depth=1,
            min_cross_leaf_score=.85,
            score_decay_per_depth=0.0,
            min_score_floor=.85,
            min_cross_leaf_pair_fraction=1.0,
            membership_confidence=.95,
        ))
        assert hierarchy_report.generated_concepts == 1, hierarchy_report.to_dict()
        persist_frame_document(db.db, hierarchy_doc); db.db.commit()

        transfer_doc, transfer_report = promote_frames_to_higher_concepts(db.db, HierarchicalFrameTransferPolicy(
            min_ancestor_hops=1,
            max_ancestor_hops=2,
            min_path_edge_quality=.50,
            hierarchical_admission_min_path_score=.70,
        ))
        assert transfer_report.frames_promoted == 2, transfer_report.to_dict()
        persist_frame_document(db.db, transfer_doc); db.db.commit()

        # Exception observations arrive *after* the broad rule and hierarchy already
        # exist. Forge must not rerun frame induction and accidentally treat their
        # target label as subtype evidence.
        for n in exc_doc.nodes: db.upsert_node(n)
        for e in exc_doc.evidence: db.add_evidence(e)
        db.commit()

        override_doc, override_report = induce_exception_overrides(db.db, ExceptionInductionPolicy(
            min_contexts=2,
            min_unique_fillers=2,
            min_target_context_fraction=1.0,
            min_exception_target_score=.70,
            min_disagreement_margin=.50,
            min_membership_quality=.80,
            min_prototype_path_quality=.50,
            max_prototype_hops=2,
        ))
        errors = [x for x in validate_document(override_doc) if x.severity == 'error']
        assert not errors, [x.message for x in errors]
        assert override_report.overrides_induced == 1, override_report.to_dict()
        binding = override_report.bindings[0]
        assert binding.pattern == ['*','atmet']
        assert binding.support_fillers == ['Adler','Falke']
        assert binding.exception_top_target_surface == 'Sauerstoff'
        assert binding.general_top_target_surface == 'Luft'
        assert binding.disagreement_margin > .65
        persist_frame_document(db.db, override_doc); db.db.commit()

        comp = _pass_through_compression(db.db)
        manifest = compile_base_pack(
            db.db, root/'exception.eqbase', pack_id='v028-exception', pack_version=28,
            compression_run_id=comp.compression_run_id, include_metadata=True,
            relation_policy=CompilePolicy(min_score=0.0, min_confidence=0.0, min_sources=1),
        )
        assert manifest.roundtrip_valid

    pack = load_base_pack(root/'exception.eqbase')
    isolated = dict(
        max_recall_nodes=64, max_recall_depth=2, max_predictions=4, min_prediction_score=0.0,
        enable_exact_context=False, enable_suffix_context=False, enable_lexical_seeds=False,
        enable_parent_routing=False, enable_relation_recall=False, enable_transition_prediction=False,
        enable_recalled_context_prediction=False, enable_direct_context_prediction=True,
        enable_frame_schema_matching=True, enable_semantic_slot_admission=True,
        enable_slot_prototype_admission=True, enable_hierarchical_slot_admission=True,
        enable_frame_specificity_override=True,
    )
    baseline = RuntimeCognition(pack, RuntimeCognitionPolicy(**isolated, enable_exception_override_frames=False))
    enabled = RuntimeCognition(pack, RuntimeCognitionPolicy(**isolated, enable_exception_override_frames=True))

    old = baseline.predict('Adler atmet')
    assert old.predictions and old.predictions[0].surface == 'Luft', [x.to_dict() for x in old.predictions]
    new = enabled.predict('Adler atmet')
    assert new.predictions and new.predictions[0].surface == 'Sauerstoff', [x.to_dict() for x in new.predictions]
    assert any(x.get('admitted') and not x.get('suppressed_by_specificity') for x in new.frame_admission)
    assert any(x.get('suppressed_by_specificity') for x in new.frame_admission)

    # Eule was never observed in an "* atmet" teacher context. It existed only in
    # the independent semantic support graph and was preclassified into the bird
    # prototype before the exception observations arrived. The learned override
    # must therefore transfer to a genuinely unseen frame instance.
    eule_old = baseline.predict('Eule atmet')
    assert eule_old.predictions and eule_old.predictions[0].surface == 'Luft'
    eule_new = enabled.predict('Eule atmet')
    assert eule_new.predictions and eule_new.predictions[0].surface == 'Sauerstoff'

    # The exception remains subtype-specific. A source-subtype member still follows
    # the broad/original rule and does not inherit the bird exception.
    hund = enabled.predict('Hund atmet')
    assert hund.predictions and hund.predictions[0].surface == 'Luft'

    report = benchmark_exception_overrides(
        pack,
        [ExceptionCase('Eule atmet','Sauerstoff','held-out frame instance'), ExceptionCase('Hund atmet','Luft','general-rule control')],
        policy=RuntimeCognitionPolicy(**isolated),
    )
    assert report.baseline_top1_accuracy == .5, report.to_dict()
    assert report.exception_top1_accuracy == 1.0, report.to_dict()
    assert report.improved_cases == 1 and report.regressed_cases == 0
    assert report.exception_frames_used == 1

print('FORGE_V28_EXCEPTION_INDUCTION_OK')
