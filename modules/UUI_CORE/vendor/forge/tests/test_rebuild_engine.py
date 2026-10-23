from pathlib import Path
import tempfile

from equinox_forge.database import CanonicalDB
from equinox_forge.ir import IRNode, RelationEvidence, RelationType, SourceRef, NodeKind, Modality
from equinox_forge.compiler import aggregate_relations
from equinox_forge.sense_compiler import SenseCompilePolicy, FusionWeights
from equinox_forge.consensus import ConsensusPolicy
from equinox_forge.rebuild import (
    effective_evidence_view,
    set_source_policy,
    replace_source_version,
    source_policy_hash,
    rebuild_senses,
    list_source_policies,
)


def add_node(db, nid, kind, level=1, surface=None, members=None):
    db.upsert_node(IRNode(nid, Modality.TEXT.value, kind, level, surface, members or [], {}))


def add_ev(db, a, b, typ, score, conf, src, ver='1', extraction='fixture'):
    db.add_evidence(RelationEvidence(a,b,typ,score,conf,SourceRef(src,ver,extraction),{}))


with tempfile.TemporaryDirectory() as td:
    path=Path(td)/'rebuild.sqlite'
    with CanonicalDB(path) as c:
        # Small sense fixture: four contexts form two clear clusters.
        add_node(c,'bank',NodeKind.SURFACE.value,1,'Bank')
        for i in range(1,5):
            add_node(c,f'ctx{i}',NodeKind.COMPOSITE.value,2,None,['bank'])
            add_ev(c,'bank',f'ctx{i}',RelationType.CONTEXTUAL_USE.value,1.0,1.0,'fixture-use')
        for src,score in [('teacher-A',.95),('teacher-B',.93)]:
            add_ev(c,'ctx1','ctx2',RelationType.CONTEXTUAL_SIMILARITY.value,score,1.0,src,extraction='hidden')
            add_ev(c,'ctx3','ctx4',RelationType.CONTEXTUAL_SIMILARITY.value,score,1.0,src,extraction='hidden')
        # Separate relation used to prove trust weighting changes normal aggregation.
        add_node(c,'x',NodeKind.SURFACE.value,1,'x'); add_node(c,'y',NodeKind.SURFACE.value,1,'y')
        add_ev(c,'x','y',RelationType.SEMANTIC_SIMILARITY.value,.9,1.0,'weight-A')
        add_ev(c,'x','y',RelationType.SEMANTIC_SIMILARITY.value,.1,1.0,'weight-B')
        c.commit()
        raw_count=c.db.execute('SELECT COUNT(*) FROM main.evidence').fetchone()[0]

        # Explicit default is behaviourally identical, therefore policy hash is stable.
        h0=source_policy_hash(c.db)
        set_source_policy(c.db,'teacher-A','1',enabled=True,trust_weight=1.0,note='no-op explicit default')
        assert source_policy_hash(c.db)==h0

        # Reweight one source: raw DB is untouched, effective aggregate moves toward A.
        set_source_policy(c.db,'weight-B','1',enabled=True,trust_weight=.1)
        with effective_evidence_view(c.db) as meta:
            rel={(r.from_id,r.to_id,r.relation_type):r for r in aggregate_relations(c.db)}
            r=rel[('x','y',RelationType.SEMANTIC_SIMILARITY.value)]
            assert r.score > .80, r
            assert meta['raw_evidence_count']==raw_count
            assert meta['effective_evidence_count']==raw_count
        assert c.db.execute('SELECT COUNT(*) FROM main.evidence').fetchone()[0]==raw_count

        # Two independent full-trust teachers -> candidate, then stable on repeated rebuild.
        sense=SenseCompilePolicy(
            threshold=.75,min_channels=1,min_contexts=2,
            weights=FusionWeights(hidden_similarity=1.0,lexical_overlap=0.0,semantic_similarity=0.0,substitution=0.0,transition=0.0,prediction_signature=0.0),
        )
        consensus=ConsensusPolicy(
            stable_min_sources=2,stable_min_runs=2,stable_min_support_weight=1.5,
            stable_consensus=.75,quarantine_conflicts=2,quarantine_conflict_weight=1.5,
        )
        d1,r1=rebuild_senses(c.db,'bank',sense,consensus)
        assert {h['status'] for h in r1.consensus_report['hypotheses']}=={'candidate'}
        d2,r2=rebuild_senses(c.db,'bank',sense,consensus)
        assert {h['status'] for h in r2.consensus_report['hypotheses']}=={'stable'}

        # Teacher B still counts as an independent source, but only contributes 0.4 trust.
        set_source_policy(c.db,'teacher-B','1',enabled=True,trust_weight=.4)
        _,r3=rebuild_senses(c.db,'bank',sense,consensus)
        hs=r3.consensus_report['hypotheses']
        assert {h['status'] for h in hs}=={'candidate'}, hs
        assert all(abs(h['support_weight']-1.4)<1e-9 for h in hs), hs
        assert all(h['support_sources']==2 for h in hs)

        # Ablating A removes its evidence from the effective view but not from raw history.
        set_source_policy(c.db,'teacher-A','1',enabled=False,trust_weight=1.0)
        _,r4=rebuild_senses(c.db,'bank',sense,consensus)
        hs=r4.consensus_report['hypotheses']
        # With only the 0.4-trust teacher left, effective hidden affinity drops below
        # the sense threshold. The current senses disappear and their persisted
        # lifecycle is downgraded as a missing rebuild result.
        assert hs == [], hs
        missing=r4.consensus_report['missing_hypotheses']
        assert len(missing)==2 and {m['status'] for m in missing}=={'contested'}, missing
        assert r4.effective_evidence_count < r4.raw_evidence_count
        assert c.db.execute('SELECT COUNT(*) FROM main.evidence').fetchone()[0]==raw_count

        # Version replacement is policy-only and atomic: old stays in raw DB but is disabled.
        old,new=replace_source_version(c.db,'teacher-B','1','2',new_trust_weight=.8)
        policies={(p['source_id'],p['source_version']):p for p in list_source_policies(c.db)}
        assert policies[('teacher-B','1')]['enabled'] is False
        assert policies[('teacher-B','2')]['enabled'] is True
        assert abs(policies[('teacher-B','2')]['trust_weight']-.8)<1e-12
        stats=c.stats()
        assert stats['rebuild_runs']==4, stats
        assert stats['source_policies']>=4, stats

print('FORGE_V09_REBUILD_ENGINE_OK')
