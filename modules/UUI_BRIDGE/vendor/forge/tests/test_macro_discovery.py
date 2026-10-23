from pathlib import Path
import tempfile

from equinox_forge.database import CanonicalDB
from equinox_forge.ir import IRNode, NodeKind, Modality
from equinox_forge.macro_discovery import MacroDiscoveryPolicy, discover_and_optionally_materialize
from equinox_forge.artifact_store import list_artifacts


def add(db, nid, level, members=None):
    db.upsert_node(IRNode(nid, Modality.TEXT.value, NodeKind.COMPOSITE.value if members else NodeKind.SURFACE.value, level, nid, members or [], {}))

with tempfile.TemporaryDirectory() as td:
    path=Path(td)/'discovery.sqlite'
    with CanonicalDB(path) as c:
        # Atomic reusable children.
        for nid in ('a','b','c','d','x','y','z'):
            add(c,nid,1)
        # The exact [a,b,c] structure occurs in three independent parents.
        add(c,'p1',2,['a','b','c','d'])
        add(c,'p2',2,['x','a','b','c','y'])
        add(c,'p3',2,['a','b','c','z'])
        # Overlap guard: [a,a] has two raw windows but only one usable replacement here.
        add(c,'p4',2,['a','a','a'])
        c.commit()

        policy=MacroDiscoveryPolicy(
            min_pattern_size=2,max_pattern_size=4,
            min_occurrences=2,min_parent_structures=2,
            definition_overhead=1.0,min_net_savings=1.0,
            min_compression_ratio=.10,prune_dominated=True,
        )
        report, run_id=discover_and_optionally_materialize(c.db,policy,materialize=False)
        assert run_id is not None
        by_members={tuple(x.members):x for x in report.candidates}
        abc=by_members[('a','b','c')]
        assert abc.eligible, abc.to_dict()
        assert abc.usable_occurrences==3
        assert abc.parent_structures==3
        assert abc.baseline_refs==9.0
        assert abc.encoded_refs==6.0
        assert abc.net_savings==2.0
        assert round(abc.compression_ratio,6)==round(2/9,6)
        # [a,b] repeats but costs as much as it saves after definition overhead.
        ab=by_members[('a','b')]
        assert not ab.eligible and ab.net_savings==0.0
        # The overlapping p4 contribution must not produce a fake 2-use candidate on its own.
        if ('a','a') in by_members:
            aa=by_members[('a','a')]
            assert aa.usable_occurrences==1
            assert not aa.eligible
        assert c.stats()['macro_discovery_runs']==1
        assert c.stats()['macro_candidates']>=3

        # Explicit materialization builds the top eligible pattern as a normal v0.11 macro artifact.
        report2, run2=discover_and_optionally_materialize(c.db,policy,materialize=True,materialize_limit=1)
        assert run2 is not None
        assert len(report2.materialized)==1
        mat=report2.materialized[0]
        assert mat['candidate_id']==abc.candidate_id
        assert mat['built']
        arts=list_artifacts(c.db,build_target_id=mat['target_id'])
        assert len(arts)==1 and arts[0]['status']=='active'
        node=(arts[0]['payload'].get('node') or {})
        assert node.get('members')==['a','b','c']
        assert mat['potential_net_savings']==2.0

print('FORGE_V12_MACRO_DISCOVERY_OK')
