from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tempfile

from equinox_forge.database import CanonicalDB
from equinox_forge.ir import IRNode, NodeKind, Modality, RelationEvidence, SourceRef, RelationType
from equinox_forge.macro_discovery import MacroDiscoveryPolicy
from equinox_forge.runtime_compression import MacroPromotionPolicy, compile_runtime_compression
from equinox_forge.runtime_pack import compile_base_pack, load_base_pack
from equinox_forge.runtime_query import RuntimeIndex, benchmark_runtime_pack, query_runtime_pack


def add(db, nid, level=1, members=None):
    db.upsert_node(IRNode(
        nid, Modality.TEXT.value,
        NodeKind.COMPOSITE.value if members else NodeKind.SURFACE.value,
        level, nid, members or [], {"fixture": "v015"},
    ))


with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    db_path = root / "bench.sqlite"
    pack_path = root / "bench.eqbase"
    parent_count = 80
    with CanonicalDB(db_path) as c:
        for nid in ("a", "b", "c"):
            add(c, nid)
        src = SourceRef("teacher-v015", "1", "runtime_benchmark_fixture")
        for i in range(parent_count):
            left = f"left-{i:03d}"
            right = f"right-{i:03d}"
            parent = f"parent-{i:03d}"
            add(c, left)
            add(c, right)
            add(c, parent, 2, [left, "a", "b", "c", right])
            c.add_evidence(RelationEvidence(
                "a", right, RelationType.PREDICTION.value,
                0.5 + (i % 10) / 100.0, 0.9, src, {"fixture_index": i},
            ))
        c.commit()

        discovery = MacroDiscoveryPolicy(
            min_pattern_size=2, max_pattern_size=3,
            min_occurrences=3, min_parent_structures=3,
            definition_overhead=1.0, min_net_savings=1.0,
            min_compression_ratio=0.0, prune_dominated=True,
        )
        promotion = MacroPromotionPolicy(
            min_realized_occurrences=3,
            min_realized_parent_structures=3,
            definition_overhead=1.0,
            min_realized_net_savings=1.0,
            min_realized_compression_ratio=0.0,
            max_promotions=8,
        )
        compression = compile_runtime_compression(c.db, discovery, promotion, persist=True, materialize=True)
        abc = next(x for x in compression.promoted if tuple(x.members) == ("a", "b", "c"))
        manifest = compile_base_pack(
            c.db, pack_path, pack_id="v015-bench", pack_version=15,
            compression_run_id=compression.compression_run_id,
            include_metadata=False, persist=False, verify_roundtrip=True,
        )
        assert manifest.roundtrip_valid

    loaded = load_base_pack(pack_path)
    idx = RuntimeIndex(loaded)
    assert idx.stats().node_count == len(loaded.nodes)
    assert idx.stats().root_count == parent_count

    parents, pm = idx.parent_nodes(abc.macro_node_id)
    assert len(parents) == parent_count
    assert pm.indexed_work_units < pm.baseline_work_units

    rels, rm = idx.relation_neighbors("a", direction="out")
    assert len(rels) == parent_count
    assert rm.indexed_work_units < rm.baseline_work_units or len(loaded.relations) == parent_count

    surface_nodes, sm = idx.lookup_surface("a")
    assert len(surface_nodes) == 1
    assert sm.reduction > 0.90

    leaves, tm = idx.expand_leaves("parent-000")
    assert leaves == ["left-000", "a", "b", "c", "right-000"]
    assert tm["leaf_count"] == 5

    report = benchmark_runtime_pack(
        loaded, sample_nodes=64, sample_macros=8,
        timing_repeat=0, max_expanded_symbols=1_000_000,
    )
    assert report.node_lookup["work_reduction"] > 0.90
    assert report.surface_lookup["work_reduction"] > 0.90
    assert report.parent_lookup["work_reduction"] > 0.90
    route = next(x for x in report.routing if x["macro_node_id"] == abc.macro_node_id)
    assert route["baseline_matching_roots"] == parent_count
    assert route["hierarchical_matching_roots"] == parent_count
    assert route["coverage_of_baseline_matches"] == 1.0
    assert route["work_reduction"] > 0.70
    assert report.routing_summary["work_reduction"] > 0.50

    query = query_runtime_pack(pack_path, parents_of=abc.macro_node_id)
    assert len(query["result"]) == parent_count
    assert query["metrics"]["indexed_work_units"] < query["metrics"]["baseline_work_units"]

print("FORGE_V15_RUNTIME_QUERY_BENCHMARK_OK")
