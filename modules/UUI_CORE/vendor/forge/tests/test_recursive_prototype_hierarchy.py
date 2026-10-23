from pathlib import Path
import tempfile, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from equinox_forge.database import CanonicalDB
from equinox_forge.teacher_behavior import TeacherCandidate, BehaviourProbeConfig, extract_teacher_behaviour
from equinox_forge.frame_induction import FrameInductionPolicy, induce_relation_frames, persist_frame_document
from equinox_forge.prototype_hierarchy import PrototypeHierarchyPolicy, induce_prototype_hierarchy
from equinox_forge.canonical import make_text_surface
from equinox_forge.ir import ForgeDocument, RelationEvidence, RelationType, SourceRef
from equinox_forge.validate import validate_document
from equinox_forge.generalization import _pass_through_compression
from equinox_forge.compiler import CompilePolicy
from equinox_forge.runtime_pack import compile_base_pack, load_base_pack
from equinox_forge.runtime_query import RuntimeIndex


class ScriptedHierarchyTeacher:
    backend_id = 'scripted-hierarchy-teacher'
    def manifest(self): return {'backend_id': self.backend_id, 'fixture': True}
    def next_token(self, prompt, top_k, temperature):
        table = {
            'Hund ist ein': [(' Tier', .90), (' Wesen', .10)],
            'Katze ist ein': [(' Tier', .90), (' Wesen', .10)],
            'Adler kann': [(' fliegen', .90), (' jagen', .10)],
            'Falke kann': [(' fliegen', .90), (' jagen', .10)],
            'Lachs lebt': [(' Wasser', .90), (' Fluss', .10)],
            'Forelle lebt': [(' Wasser', .90), (' Fluss', .10)],
            'Eiche braucht': [(' Wasser', .90), (' Licht', .10)],
            'Buche braucht': [(' Wasser', .90), (' Licht', .10)],
        }
        vals = table[prompt.strip()][:top_k]
        return [TeacherCandidate(i + 1, s, s, prob, 3.0 - i) for i, (s, prob) in enumerate(vals)]


def support_doc():
    names = ['Hund','Katze','Adler','Falke','Lachs','Forelle','Eiche','Buche']
    nodes = {}; surfaces = {}
    for text in names:
        qs, node = make_text_surface(text)
        for q in qs: nodes[q.node_id] = q
        nodes[node.node_id] = node
        surfaces[text] = node
    src = SourceRef('independent-semantic-hierarchy', '1', 'semantic_similarity_fixture')
    pairs = []
    groups = [
        ['Hund','Katze'], ['Adler','Falke'], ['Lachs','Forelle'], ['Eiche','Buche']
    ]
    # Within each base prototype: strong enough for v0.24 slot typing.
    for a, b in groups:
        pairs.append((a, b, .98))
    # First recursive layer: prototype 0<->1 and 2<->3.
    for ga, gb in ((groups[0], groups[1]), (groups[2], groups[3])):
        for a in ga:
            for b in gb:
                pairs.append((a, b, .90))
    # Second recursive layer: broader relation, deliberately below depth-1 threshold
    # but above the relaxed depth-2 threshold.
    for a in groups[0] + groups[1]:
        for b in groups[2] + groups[3]:
            pairs.append((a, b, .80))
    ev = [
        RelationEvidence(surfaces[a].node_id, surfaces[b].node_id, RelationType.SEMANTIC_SIMILARITY.value,
                         score, .99, src, {'independent_of_teacher': True})
        for a, b, score in pairs
    ]
    return ForgeDocument(nodes=sorted(nodes.values(), key=lambda n:(n.level,n.node_id)), evidence=ev)


with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    db_path = root / 'master.sqlite'
    teacher_doc, _, _ = extract_teacher_behaviour(
        ScriptedHierarchyTeacher(),
        ['Hund ist ein','Katze ist ein','Adler kann','Falke kann','Lachs lebt','Forelle lebt','Eiche braucht','Buche braucht'],
        'teacher-hierarchy', '1', BehaviourProbeConfig(top_k=2, min_probability=0.0),
    )
    semantic = support_doc()
    with CanonicalDB(db_path) as db:
        for doc in (teacher_doc, semantic):
            for n in doc.nodes: db.upsert_node(n)
            for e in doc.evidence: db.add_evidence(e)
        db.commit()

        frame_policy = FrameInductionPolicy(
            enable_semantic_slot_typing=True,
            min_filler_coherence_score=.95,
            min_filler_pair_fraction=1.0,
            require_typed_slot=True,
            enable_latent_slot_prototype=True,
            prototype_classify_candidates=False,
            prototype_min_admission_score=.95,
            prototype_min_support_fillers=2,
        )
        frame_doc, frame_report = induce_relation_frames(db.db, frame_policy)
        assert frame_report.generated_slot_prototypes == 4, frame_report.to_dict()
        assert not [x for x in validate_document(frame_doc) if x.severity == 'error']
        persist_frame_document(db.db, frame_doc)
        db.db.commit()

        hierarchy_policy = PrototypeHierarchyPolicy(
            max_depth=4,
            min_cross_leaf_score=.88,
            score_decay_per_depth=.10,
            min_score_floor=.68,
            min_cross_leaf_pair_fraction=1.0,
            membership_confidence=.95,
        )
        hierarchy_doc, report = induce_prototype_hierarchy(db.db, hierarchy_policy)
        errors = [x for x in validate_document(hierarchy_doc) if x.severity == 'error']
        assert not errors, [x.message for x in errors]
        assert report.base_prototypes == 4
        assert report.generated_concepts == 3, report.to_dict()  # two parents + one root
        assert report.generated_memberships == 6
        assert report.highest_hierarchy_depth == 2
        assert report.levels_generated == 2
        assert sorted(m.hierarchy_depth for m in report.merges) == [1,1,2]
        depth1 = [m for m in report.merges if m.hierarchy_depth == 1]
        depth2 = [m for m in report.merges if m.hierarchy_depth == 2]
        assert len(depth1) == 2 and len(depth2) == 1
        assert all(abs(m.threshold - .88) < 1e-9 for m in depth1)
        assert abs(depth2[0].threshold - .78) < 1e-9
        root_id = depth2[0].node_id
        root_node = next(n for n in hierarchy_doc.nodes if n.node_id == root_id)
        assert (root_node.metadata or {}).get('role') == 'recursive_latent_concept'
        assert len(root_node.members) == 2
        assert len((root_node.metadata or {}).get('leaf_node_ids', [])) == 8

        persist_frame_document(db.db, hierarchy_doc)
        db.db.commit()
        comp = _pass_through_compression(db.db)
        manifest = compile_base_pack(
            db.db, root/'hierarchy.eqbase', pack_id='v025-hierarchy', pack_version=25,
            compression_run_id=comp.compression_run_id, include_metadata=True,
            relation_policy=CompilePolicy(min_score=0.0, min_confidence=0.0, min_sources=1),
        )
        assert manifest.roundtrip_valid

    pack = load_base_pack(root/'hierarchy.eqbase')
    idx = RuntimeIndex(pack)
    base_protos = [n for n in pack.nodes if (n.metadata or {}).get('role') == 'latent_slot_prototype']
    recursive = [n for n in pack.nodes if (n.metadata or {}).get('role') == 'recursive_latent_concept']
    assert len(base_protos) == 4
    assert len(recursive) == 3
    # Every base concept must reach the same recursive root through IS_A only.
    for proto in base_protos:
        reachable, _ = idx.traverse_relation(proto.node_id, RelationType.IS_A.value, max_depth=4)
        ids = {row['node_id'] for row in reachable}
        assert root_id in ids, (proto.surface, ids, root_id)
    root = idx.node_by_id[root_id]
    assert int((root.metadata or {}).get('hierarchy_depth', 0)) == 2

print('FORGE_V25_RECURSIVE_PROTOTYPE_HIERARCHY_OK')
