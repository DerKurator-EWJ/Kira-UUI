from pathlib import Path
import tempfile

from equinox_forge.database import CanonicalDB
from equinox_forge.ir import IRNode, RelationEvidence, RelationType, SourceRef, NodeKind, Modality
from equinox_forge.sense_compiler import SenseCompilePolicy, FusionWeights
from equinox_forge.consensus import ConsensusPolicy
from equinox_forge.rebuild import rebuild_senses, set_source_policy
from equinox_forge.macro_builder import MacroCompilePolicy, rebuild_macro
from equinox_forge.artifact_store import list_artifacts, current_output_nodes
from equinox_forge.incremental import incremental_rebuild_from_sources
from equinox_forge.validate import validate_document
from equinox_forge.dependency import refresh_dependency_index


def add_node(db, nid, kind, level=1, surface=None, members=None):
    db.upsert_node(IRNode(nid, Modality.TEXT.value, kind, level, surface, members or [], {}))


def add_ev(db, a, b, typ, score, conf, src, ver='1', extraction='fixture'):
    db.add_evidence(RelationEvidence(a,b,typ,score,conf,SourceRef(src,ver,extraction),{}))


def add_target(c, target, teacher):
    contexts=[f'{target}-ctx1',f'{target}-ctx2']
    add_node(c,f'{target}-q',NodeKind.QUANT.value,0,target[:1])
    add_node(c,target,NodeKind.SURFACE.value,1,target.title(),[f'{target}-q'])
    for ctx in contexts:
        add_node(c,ctx,NodeKind.COMPOSITE.value,2,None,[target])
        add_ev(c,target,ctx,RelationType.CONTEXTUAL_USE.value,1.0,1.0,f'use-{target}')
    add_ev(c,contexts[0],contexts[1],RelationType.CONTEXTUAL_SIMILARITY.value,.96,1.0,teacher,extraction='hidden')


with tempfile.TemporaryDirectory() as td:
    path=Path(td)/'hierarchy.sqlite'
    out=Path(td)/'out'
    with CanonicalDB(path) as c:
        add_target(c,'bank','teacher-bank')
        add_target(c,'kran','teacher-kran')
        add_target(c,'hund','teacher-hund')
        c.commit()

        sense=SenseCompilePolicy(
            threshold=.75,min_channels=1,min_contexts=2,
            weights=FusionWeights(hidden_similarity=1.0,lexical_overlap=0.0,semantic_similarity=0.0,substitution=0.0,transition=0.0,prediction_signature=0.0),
        )
        consensus=ConsensusPolicy(
            stable_min_sources=1,stable_min_runs=1,stable_consensus=.5,
            stable_min_support_weight=.5,quarantine_conflicts=2,quarantine_conflict_weight=1.5,
        )

        # Three stable first-level derived senses are persisted in the Artifact Store.
        for target in ('bank','kran','hund'):
            rebuild_senses(c.db,target,sense,consensus)
            arts=list_artifacts(c.db,build_target_id=target)
            assert len(arts)==1 and arts[0]['status']=='stable', (target,arts)

        # First higher macro consumes two stable sense targets without requiring the
        # sense nodes to be copied into canonical input tables.
        pair_policy=MacroCompilePolicy(
            input_build_targets=['bank','kran'],allowed_statuses=['stable'],min_members=2,
            surface='macro:bank+kran',
        )
        pair_doc,pair_rep=rebuild_macro(c.db,'macro-pair',pair_policy)
        assert not [x for x in validate_document(pair_doc) if x.severity=='error']
        assert pair_rep.built and len(pair_rep.members)==2
        assert pair_rep.level is not None and pair_rep.level>=4
        pair_node=pair_rep.output_node_id
        assert current_output_nodes(c.db,'macro-pair',['active'])==[(pair_node,'active')]
        # It really came from the artifact store: the derived sense children were not
        # force-ingested into the canonical nodes table.
        derived_children=pair_rep.members
        assert all(c.db.execute('SELECT 1 FROM nodes WHERE node_id=?',(nid,)).fetchone() is None for nid in derived_children)

        # A second macro recursively consumes macro-pair plus a third sense.
        top_policy=MacroCompilePolicy(
            input_build_targets=['macro-pair','hund'],allowed_statuses=['active','stable'],min_members=2,
            surface='macro:domain',
        )
        top_doc,top_rep=rebuild_macro(c.db,'macro-domain',top_policy)
        assert not [x for x in validate_document(top_doc) if x.severity=='error']
        assert top_rep.built and len(top_rep.members)==2
        assert pair_node in top_rep.members
        assert top_rep.level > pair_rep.level

        # Rebuilding an identical macro is idempotent in version history.
        versions_before=c.db.execute("SELECT COUNT(*) FROM derived_artifact_versions WHERE build_target_id='macro-pair'").fetchone()[0]
        rebuild_macro(c.db,'macro-pair',pair_policy)
        versions_after=c.db.execute("SELECT COUNT(*) FROM derived_artifact_versions WHERE build_target_id='macro-pair'").fetchone()[0]
        assert versions_before==versions_after

        refresh_dependency_index(c.db)

        # Remove only bank's teacher support. Incremental invalidation must flow:
        # teacher-bank -> bank sense -> macro-pair -> macro-domain.
        set_source_policy(c.db,'teacher-bank','1',enabled=False,trust_weight=0.0)
        rep=incremental_rebuild_from_sources(c.db,['teacher-bank@1'],output_dir=out)
        assert rep.affected_build_targets==['bank','macro-domain','macro-pair'], rep.to_dict()
        order=[x['target_id'] for x in rep.rebuilt_targets]
        assert order==['bank','macro-pair','macro-domain'], order
        states={x['target_id']:x['status'] for x in rep.rebuilt_targets}
        assert states['bank']=='rebuilt'
        assert states['macro-pair']=='not_built'
        assert states['macro-domain']=='not_built'

        # Only the affected hierarchy goes stale. Independent senses remain stable.
        assert list_artifacts(c.db,build_target_id='bank')[0]['status']=='stale'
        assert list_artifacts(c.db,build_target_id='macro-pair')[0]['status']=='stale'
        assert list_artifacts(c.db,build_target_id='macro-domain')[0]['status']=='stale'
        assert list_artifacts(c.db,build_target_id='kran')[0]['status']=='stable'
        assert list_artifacts(c.db,build_target_id='hund')[0]['status']=='stable'
        assert c.stats()['derived_artifact_versions'] >= 8

print('FORGE_V11_DERIVED_MACRO_HIERARCHY_OK')
