from pathlib import Path
import tempfile

from equinox_forge.database import CanonicalDB
from equinox_forge.ir import IRNode, RelationEvidence, RelationType, SourceRef, NodeKind, Modality
from equinox_forge.sense_compiler import SenseCompilePolicy, FusionWeights
from equinox_forge.consensus import ConsensusPolicy
from equinox_forge.rebuild import rebuild_senses, set_source_policy
from equinox_forge.dependency import (
    ArtifactRef, impact_from_roots, register_manual_dependency, refresh_dependency_index, register_build_target
)
from equinox_forge.incremental import incremental_rebuild_from_sources


def add_node(db, nid, kind, level=1, surface=None, members=None):
    db.upsert_node(IRNode(nid, Modality.TEXT.value, kind, level, surface, members or [], {}))


def add_ev(db, a, b, typ, score, conf, src, ver='1', extraction='fixture'):
    db.add_evidence(RelationEvidence(a,b,typ,score,conf,SourceRef(src,ver,extraction),{}))


def add_target_fixture(c, target, contexts, teacher):
    add_node(c,target,NodeKind.SURFACE.value,1,target.title())
    for ctx in contexts:
        add_node(c,ctx,NodeKind.COMPOSITE.value,2,None,[target])
        add_ev(c,target,ctx,RelationType.CONTEXTUAL_USE.value,1.0,1.0,f'use-{target}')
    add_ev(c,contexts[0],contexts[1],RelationType.CONTEXTUAL_SIMILARITY.value,.95,1.0,teacher,extraction='hidden')
    add_ev(c,contexts[2],contexts[3],RelationType.CONTEXTUAL_SIMILARITY.value,.94,1.0,teacher,extraction='hidden')


with tempfile.TemporaryDirectory() as td:
    path=Path(td)/'incremental.sqlite'
    out=Path(td)/'out'
    with CanonicalDB(path) as c:
        add_target_fixture(c,'bank',['b1','b2','b3','b4'],'teacher-bank')
        add_target_fixture(c,'kran',['k1','k2','k3','k4'],'teacher-kran')
        c.commit()

        sense=SenseCompilePolicy(
            threshold=.75,min_channels=1,min_contexts=2,
            weights=FusionWeights(hidden_similarity=1.0,lexical_overlap=0.0,semantic_similarity=0.0,substitution=0.0,transition=0.0,prediction_signature=0.0),
        )
        consensus=ConsensusPolicy(stable_min_sources=1,stable_min_runs=1,stable_min_support_weight=.5)

        # v0.10 rebuild records a persistent builder recipe for each target.
        rebuild_senses(c.db,'bank',sense,consensus)
        rebuild_senses(c.db,'kran',sense,consensus)
        assert c.stats()['build_targets']==2

        # Generic higher artifacts prove transitive propagation beyond the built-in sense compiler.
        register_manual_dependency(c.db,'artifact','macro-bank','build_target','bank','manual_macro_dependency')
        register_manual_dependency(c.db,'artifact','domain-bank','artifact','macro-bank','manual_domain_dependency')
        register_build_target(c.db,'future-macro-bank','macro',{'note':'future compiler placeholder'})
        register_manual_dependency(c.db,'build_target','future-macro-bank','build_target','bank','future_builder_dependency')
        refresh_dependency_index(c.db)

        impact=impact_from_roots(c.db,[ArtifactRef('source','teacher-bank@1')],refresh=False)
        assert impact.affected_build_targets==['bank','future-macro-bank'], impact.to_dict()
        assert 'artifact:macro-bank' in impact.affected_manual_artifacts
        assert 'artifact:domain-bank' in impact.affected_manual_artifacts
        assert 'kran' not in impact.affected_build_targets

        before=c.db.execute('SELECT COUNT(*) FROM rebuild_runs WHERE target_id="kran"').fetchone()[0]
        set_source_policy(c.db,'teacher-bank','1',enabled=True,trust_weight=.5)
        rep=incremental_rebuild_from_sources(c.db,['teacher-bank@1'],output_dir=out)
        assert rep.affected_build_targets==['bank','future-macro-bank'], rep.to_dict()
        assert len(rep.rebuilt_targets)==1 and rep.rebuilt_targets[0]['target_id']=='bank'
        assert rep.deferred_targets==[{'target_id':'future-macro-bank','builder':'macro','reason':'builder_not_installed'}]
        assert 'artifact:domain-bank' in rep.affected_manual_artifacts
        after=c.db.execute('SELECT COUNT(*) FROM rebuild_runs WHERE target_id="kran"').fetchone()[0]
        assert before==after, (before,after)
        assert c.stats()['incremental_rebuild_runs']==1
        assert list(out.glob('*.eqir.json'))

        # Dry run never writes an incremental history row and still plans only the affected target.
        count=c.stats()['incremental_rebuild_runs']
        dry=incremental_rebuild_from_sources(c.db,['teacher-kran@1'],dry_run=True)
        assert dry.affected_build_targets==['kran']
        assert dry.rebuilt_targets==[{'target_id':'kran','builder':'sense','status':'planned'}]
        assert c.stats()['incremental_rebuild_runs']==count

print('FORGE_V10_INCREMENTAL_DEPENDENCY_OK')
