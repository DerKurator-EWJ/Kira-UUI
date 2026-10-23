from pathlib import Path
import tempfile, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from equinox_forge.database import CanonicalDB
from equinox_forge.teacher_behavior import TeacherCandidate, BehaviourProbeConfig, extract_teacher_behaviour
from equinox_forge.frame_induction import FrameInductionPolicy, induce_relation_frames, persist_frame_document
from equinox_forge.prototype_hierarchy import PrototypeHierarchyPolicy, induce_prototype_hierarchy
from equinox_forge.hierarchical_frame_transfer import HierarchicalFrameTransferPolicy, promote_frames_to_higher_concepts
from equinox_forge.canonical import make_text_surface
from equinox_forge.ir import ForgeDocument, RelationEvidence, RelationType, SourceRef
from equinox_forge.validate import validate_document
from equinox_forge.generalization import _pass_through_compression
from equinox_forge.compiler import CompilePolicy
from equinox_forge.runtime_pack import compile_base_pack, load_base_pack
from equinox_forge.runtime_cognition import RuntimeCognition, RuntimeCognitionPolicy


class ScriptedTransferTeacher:
    backend_id = 'scripted-hierarchical-transfer-teacher'
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
    src = SourceRef('independent-semantic-transfer', '1', 'semantic_similarity_fixture')
    pairs = [
        ('Hund','Katze',.98),
        ('Adler','Falke',.98),
        ('Hund','Adler',.91),('Hund','Falke',.90),
        ('Katze','Adler',.90),('Katze','Falke',.91),
        ('Stein','Hund',.05),('Stein','Katze',.04),('Stein','Adler',.03),('Stein','Falke',.04),
    ]
    ev = [RelationEvidence(
        surfaces[a].node_id, surfaces[b].node_id, RelationType.SEMANTIC_SIMILARITY.value,
        score, .99, src, {'independent_of_teacher': True}
    ) for a,b,score in pairs]
    return ForgeDocument(nodes=sorted(nodes.values(), key=lambda n:(n.level,n.node_id)), evidence=ev)


with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    db_path = root/'master.sqlite'
    teacher_doc, _, _ = extract_teacher_behaviour(
        ScriptedTransferTeacher(),
        ['Hund atmet','Katze atmet','Adler fliegt','Falke fliegt'],
        'teacher-transfer', '1', BehaviourProbeConfig(top_k=2, min_probability=0.0),
    )
    support = semantic_support_doc()
    with CanonicalDB(db_path) as db:
        for doc in (teacher_doc, support):
            for n in doc.nodes: db.upsert_node(n)
            for e in doc.evidence: db.add_evidence(e)
        db.commit()

        frame_doc, frame_report = induce_relation_frames(db.db, FrameInductionPolicy(
            enable_semantic_slot_typing=True,
            min_filler_coherence_score=.95,
            min_filler_pair_fraction=1.0,
            require_typed_slot=True,
            enable_latent_slot_prototype=True,
            prototype_classify_candidates=False,
            prototype_min_admission_score=.95,
            prototype_min_support_fillers=2,
        ))
        assert frame_report.generated_slot_prototypes == 2, frame_report.to_dict()
        assert not [x for x in validate_document(frame_doc) if x.severity == 'error']
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
        assert hierarchy_report.generated_memberships == 2
        root_id = hierarchy_report.merges[0].node_id
        persist_frame_document(db.db, hierarchy_doc); db.db.commit()

        transfer_doc, transfer_report = promote_frames_to_higher_concepts(db.db, HierarchicalFrameTransferPolicy(
            min_ancestor_hops=1,
            max_ancestor_hops=2,
            min_path_edge_quality=.50,
            hierarchical_admission_min_path_score=.70,
        ))
        errors = [x for x in validate_document(transfer_doc) if x.severity == 'error']
        assert not errors, [x.message for x in errors]
        assert transfer_report.frames_promoted == 2, transfer_report.to_dict()
        assert {b.higher_concept_node_id for b in transfer_report.bindings} == {root_id}
        assert all(b.ancestor_hops == 1 for b in transfer_report.bindings)
        assert all(b.prediction_targets for b in transfer_report.bindings)
        persist_frame_document(db.db, transfer_doc); db.db.commit()

        comp = _pass_through_compression(db.db)
        manifest = compile_base_pack(
            db.db, root/'transfer.eqbase', pack_id='v026-transfer', pack_version=26,
            compression_run_id=comp.compression_run_id, include_metadata=True,
            relation_policy=CompilePolicy(min_score=0.0, min_confidence=0.0, min_sources=1),
        )
        assert manifest.roundtrip_valid

    pack = load_base_pack(root/'transfer.eqbase')
    isolated = dict(
        max_recall_nodes=64, max_recall_depth=2, max_predictions=3, min_prediction_score=0.0,
        enable_exact_context=False, enable_suffix_context=False, enable_lexical_seeds=False,
        enable_parent_routing=False, enable_relation_recall=False, enable_transition_prediction=False,
        enable_recalled_context_prediction=False, enable_direct_context_prediction=True,
        enable_frame_schema_matching=True, enable_semantic_slot_admission=True,
        enable_slot_prototype_admission=True,
    )
    old = RuntimeCognition(pack, RuntimeCognitionPolicy(**isolated, enable_hierarchical_slot_admission=False))
    new = RuntimeCognition(pack, RuntimeCognitionPolicy(**isolated, enable_hierarchical_slot_admission=True))

    # The old direct-prototype rule cannot use a frame learned only from Hund/Katze
    # for the independently learned Adler/Falke subtype.
    assert old.predict('Adler atmet').predictions == []
    assert old.predict('Falke atmet').predictions == []

    # v0.26: Adler/Falke -> their own base prototype -> higher recursive concept ->
    # promoted frame type. No 'Adler atmet' or 'Falke atmet' context was compiled.
    adler = new.predict('Adler atmet')
    falke = new.predict('Falke atmet')
    assert adler.predictions and adler.predictions[0].surface == 'Luft'
    assert falke.predictions and falke.predictions[0].surface == 'Luft'
    for report in (adler, falke):
        checks = [sc for row in report.frame_admission for sc in row.get('slot_checks', [])]
        hits = [sc for sc in checks if sc.get('reason') == 'hierarchical_prototype_membership']
        assert hits, checks
        assert any(sc.get('hierarchical_prototype_node_id') == root_id for sc in hits)
        assert any(int(sc.get('hierarchical_hops', 0)) == 2 for sc in hits)

    # A known lexical surface outside the induced hierarchy must still be rejected.
    stone = new.predict('Stein atmet')
    assert stone.predictions == []
    checks = [sc for row in stone.frame_admission for sc in row.get('slot_checks', [])]
    assert not any(sc.get('reason') == 'hierarchical_prototype_membership' for sc in checks)

print('FORGE_V26_HIERARCHICAL_FRAME_TRANSFER_OK')
