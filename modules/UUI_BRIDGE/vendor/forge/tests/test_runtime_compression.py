from pathlib import Path
import tempfile

from equinox_forge.database import CanonicalDB
from equinox_forge.ir import IRNode, NodeKind, Modality
from equinox_forge.macro_discovery import MacroDiscoveryPolicy
from equinox_forge.runtime_compression import MacroPromotionPolicy, compile_runtime_compression
from equinox_forge.artifact_store import list_artifacts


def add(db, nid, level, members=None):
    db.upsert_node(IRNode(
        nid, Modality.TEXT.value,
        NodeKind.COMPOSITE.value if members else NodeKind.SURFACE.value,
        level, nid, members or [], {},
    ))


with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "compression.sqlite"
    with CanonicalDB(path) as c:
        for nid in ("a","b","c","d","e","f","x","y","q","r","m","n","o","u","v","w"):
            add(c, nid, 1)

        # [a,b,c] occurs four times => realized net 4 under overhead=1.
        add(c, "p1", 2, ["a","b","c","d","e"])
        add(c, "p2", 2, ["a","b","c","d","f"])
        add(c, "p3", 2, ["a","b","c","x"])
        add(c, "p4", 2, ["y","a","b","c","q"])
        # [b,c,d] overlaps the strongest candidate in p1/p2; after [a,b,c]
        # promotion it has only this independent use left and must not be promoted.
        add(c, "p5", 2, ["r","b","c","d","y"])

        # Independent second useful macro [m,n,o] occurs three times => net 2.
        add(c, "r1", 2, ["m","n","o","u"])
        add(c, "r2", 2, ["m","n","o","v"])
        add(c, "r3", 2, ["w","m","n","o"])
        c.commit()

        discovery = MacroDiscoveryPolicy(
            min_pattern_size=2, max_pattern_size=4,
            min_occurrences=2, min_parent_structures=2,
            definition_overhead=1.0, min_net_savings=1.0,
            min_compression_ratio=0.0, prune_dominated=False,
        )
        promotion = MacroPromotionPolicy(
            min_realized_occurrences=2,
            min_realized_parent_structures=2,
            definition_overhead=1.0,
            min_realized_net_savings=1.0,
            min_realized_compression_ratio=0.0,
            max_promotions=8,
        )
        rep = compile_runtime_compression(c.db, discovery, promotion, persist=True, materialize=True)

        by_members = {tuple(x.members): x for x in rep.promoted}
        assert ("a","b","c") in by_members, rep.to_dict()
        assert ("m","n","o") in by_members, rep.to_dict()
        abc = by_members[("a","b","c")]
        mno = by_members[("m","n","o")]
        assert abc.realized_occurrences == 4
        assert abc.realized_net_savings == 4.0
        assert mno.realized_occurrences == 3
        assert mno.realized_net_savings == 2.0
        assert ("b","c","d") not in by_members

        # Parent member references: 36 baseline, 14 replaced-member refs removed.
        # Two macro definitions cost 3+3 refs and two definition overhead units.
        assert rep.baseline_parent_refs == 36, rep.to_dict()
        assert rep.rewritten_parent_refs == 22, rep.to_dict()
        assert rep.parent_ref_savings == 14
        assert rep.macro_definition_refs == 6.0
        assert rep.definition_overhead_cost == 2.0
        assert rep.encoded_total_cost == 30.0
        assert rep.realized_net_savings == 6.0
        assert round(rep.realized_compression_ratio, 8) == round(6/36, 8)
        assert rep.affected_parents == 7

        rw = {x.parent_id: x for x in rep.rewrites}
        assert rw["p1"].rewritten_members[0] == abc.macro_node_id
        assert rw["p1"].rewritten_members[1:] == ["d","e"]
        assert rw["p5"].rewritten_members == ["r","b","c","d","y"]
        assert rw["r3"].rewritten_members == ["w",mno.macro_node_id]

        # Promoted macro definitions are ordinary Derived Artifacts and therefore
        # reusable by later hierarchy/runtime compilation.
        for m in rep.promoted:
            arts = list_artifacts(c.db, build_target_id=m.target_id)
            assert len(arts) == 1 and arts[0]["node_id"] == m.macro_node_id

        assert c.stats()["runtime_compression_runs"] == 1
        assert c.stats()["runtime_promoted_macros"] == rep.promoted_count
        assert c.stats()["runtime_parent_rewrites"] == rep.scanned_structures

        # Planning mode must derive the same compression accounting without adding
        # another macro Artifact generation. It may persist a plan if requested.
        before_artifacts = c.stats()["derived_artifacts"]
        plan = compile_runtime_compression(c.db, discovery, promotion, persist=False, materialize=False)
        assert plan.realized_net_savings == 6.0
        assert c.stats()["derived_artifacts"] == before_artifacts

print("FORGE_V13_RUNTIME_COMPRESSION_OK")
