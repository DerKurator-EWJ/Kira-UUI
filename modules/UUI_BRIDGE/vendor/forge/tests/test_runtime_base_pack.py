from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tempfile

from equinox_forge.database import CanonicalDB
from equinox_forge.ir import IRNode, NodeKind, Modality, RelationEvidence, SourceRef, RelationType
from equinox_forge.macro_discovery import MacroDiscoveryPolicy
from equinox_forge.runtime_compression import MacroPromotionPolicy, compile_runtime_compression
from equinox_forge.runtime_pack import compile_base_pack, load_base_pack, verify_base_pack, PackFormatError


def add(db, nid, level, members=None, metadata=None):
    db.upsert_node(IRNode(
        nid, Modality.TEXT.value,
        NodeKind.COMPOSITE.value if members else NodeKind.SURFACE.value,
        level, nid, members or [], metadata or {},
    ))


with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    db_path = root / "pack.sqlite"
    pack_path = root / "test.eqbase"
    with CanonicalDB(db_path) as c:
        for nid in ("a","b","c","d","e","f","x","y","q","m","n","o","u","v","w"):
            add(c, nid, 1, metadata={"fixture": True, "name": nid})
        add(c, "p1", 2, ["a","b","c","d","e"])
        add(c, "p2", 2, ["a","b","c","d","f"])
        add(c, "p3", 2, ["a","b","c","x"])
        add(c, "p4", 2, ["y","a","b","c","q"])
        add(c, "r1", 2, ["m","n","o","u"])
        add(c, "r2", 2, ["m","n","o","v"])
        add(c, "r3", 2, ["w","m","n","o"])
        src = SourceRef("teacher-fixture", "1", "test")
        c.add_evidence(RelationEvidence("a", "b", RelationType.PREDICTION.value, 0.812345678, 0.91, src, {}))
        c.add_evidence(RelationEvidence("m", "n", RelationType.SEMANTIC_SIMILARITY.value, 0.734567891, 0.77, src, {}))
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
        compression = compile_runtime_compression(c.db, discovery, promotion, persist=True, materialize=True)
        assert compression.compression_run_id is not None

        manifest = compile_base_pack(
            c.db, pack_path, pack_id="fixture-base", pack_version=7,
            compression_run_id=compression.compression_run_id,
            include_metadata=True, persist=True, verify_roundtrip=True,
        )
        assert manifest.roundtrip_valid
        assert manifest.pack_id == "fixture-base"
        assert manifest.pack_version == 7
        assert manifest.compression_run_id == compression.compression_run_id
        assert manifest.file_bytes == pack_path.stat().st_size
        assert manifest.index_bytes == manifest.node_count * 4
        assert manifest.node_count > 0
        assert manifest.relation_count == 2
        assert c.stats()["runtime_base_pack_runs"] == 1

        loaded = load_base_pack(pack_path)
        by_id = loaded.node_by_id()
        abc = next(x for x in compression.promoted if tuple(x.members) == ("a","b","c"))
        mno = next(x for x in compression.promoted if tuple(x.members) == ("m","n","o"))
        assert by_id["p1"].members == [abc.macro_node_id, "d", "e"]
        assert by_id["r3"].members == ["w", mno.macro_node_id]
        assert by_id[abc.macro_node_id].members == ["a","b","c"]
        assert by_id["a"].metadata == {"fixture": True, "name": "a"}
        assert len(loaded.relations) == 2

        verified = verify_base_pack(pack_path)
        assert verified["valid"]
        assert not verified["dangling_members"]
        assert not verified["dangling_relation_endpoints"]

        # The same DB/run/pack metadata must compile byte-for-byte deterministically.
        pack2 = root / "test2.eqbase"
        manifest2 = compile_base_pack(
            c.db, pack2, pack_id="fixture-base", pack_version=7,
            compression_run_id=compression.compression_run_id,
            include_metadata=True, persist=False, verify_roundtrip=True,
        )
        assert pack_path.read_bytes() == pack2.read_bytes()
        assert manifest.full_file_sha256 == manifest2.full_file_sha256

    # Hash validation must reject payload corruption.
    bad = root / "bad.eqbase"
    raw = bytearray(pack_path.read_bytes())
    raw[-1] ^= 0x01
    bad.write_bytes(raw)
    try:
        load_base_pack(bad, verify_hash=True)
        raise AssertionError("corrupt pack was accepted")
    except PackFormatError as exc:
        assert "SHA-256" in str(exc)

print("FORGE_V14_RUNTIME_BASE_PACK_OK")
