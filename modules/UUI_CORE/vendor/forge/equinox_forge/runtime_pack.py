from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import hashlib
import io
import json
import sqlite3
import struct
import time

from .compiler import aggregate_relations, CompilePolicy
from .ir import IRNode, CompiledRelation
from .runtime_compression import ensure_runtime_compression_schema

PACK_FORMAT = "equinox-runtime-base-v1"
PACK_VERSION = "0.26"
MAGIC = b"EQXBASE1"
BINARY_MAJOR = 1
BINARY_MINOR = 0
# magic, major, minor, header_size, pack_version, compression_run_id,
# node_count, relation_count, string_count, pack_id_string_id,
# strings_offset, nodes_offset, relations_offset, index_offset, file_size, payload_sha256
HEADER = struct.Struct("<8sHHIIIIIIIQQQQQ32s")


@dataclass
class RuntimeNode:
    node_id: str
    modality: str
    kind: str
    level: int
    surface: str | None
    members: list[str]
    metadata: dict[str, Any]


@dataclass
class RuntimePackManifest:
    format: str
    forge_version: str
    binary_major: int
    binary_minor: int
    pack_id: str
    pack_version: int
    compression_run_id: int
    node_count: int
    relation_count: int
    string_count: int
    file_bytes: int
    header_bytes: int
    string_table_bytes: int
    node_section_bytes: int
    relation_section_bytes: int
    index_bytes: int
    payload_sha256: str
    full_file_sha256: str
    bytes_per_node: float
    bytes_per_relation: float
    compile_ms: float
    verify_ms: float
    roundtrip_valid: bool
    metadata_included: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LoadedRuntimePack:
    manifest: dict[str, Any]
    nodes: list[RuntimeNode]
    relations: list[CompiledRelation]
    node_offsets: list[int]

    def node_by_id(self) -> dict[str, RuntimeNode]:
        return {n.node_id: n for n in self.nodes}


class PackFormatError(ValueError):
    pass


def ensure_runtime_pack_schema(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS runtime_base_pack_runs(
          base_pack_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
          compression_run_id INTEGER NOT NULL,
          pack_id TEXT NOT NULL,
          pack_version INTEGER NOT NULL,
          output_path TEXT NOT NULL,
          file_bytes INTEGER NOT NULL,
          node_count INTEGER NOT NULL,
          relation_count INTEGER NOT NULL,
          string_count INTEGER NOT NULL,
          payload_sha256 TEXT NOT NULL,
          full_file_sha256 TEXT NOT NULL,
          roundtrip_valid INTEGER NOT NULL,
          manifest_json TEXT NOT NULL DEFAULT '{}',
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_runtime_base_pack_run
          ON runtime_base_pack_runs(compression_run_id,base_pack_run_id);
        """
    )


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _uvarint(value: int) -> bytes:
    if value < 0:
        raise ValueError("uvarint cannot encode a negative value")
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _read_uvarint(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    shift = 0
    for _ in range(10):
        if pos >= len(data):
            raise PackFormatError("truncated varint")
        b = data[pos]
        pos += 1
        value |= (b & 0x7F) << shift
        if not b & 0x80:
            return value, pos
        shift += 7
    raise PackFormatError("varint too long")


def _latest_compression_run(db: sqlite3.Connection) -> int:
    ensure_runtime_compression_schema(db)
    row = db.execute("SELECT MAX(compression_run_id) FROM runtime_compression_runs").fetchone()
    if not row or row[0] is None:
        raise ValueError("no persisted runtime compression run; run compress-runtime first")
    return int(row[0])


def _node_from_canonical(db: sqlite3.Connection, node_id: str) -> RuntimeNode | None:
    row = db.execute(
        "SELECT node_id,modality,kind,level,surface,metadata_json FROM nodes WHERE node_id=?", (node_id,)
    ).fetchone()
    if not row:
        return None
    members = [r[0] for r in db.execute(
        "SELECT child_id FROM node_members WHERE parent_id=? ORDER BY position", (node_id,)
    ).fetchall()]
    return RuntimeNode(row[0], row[1], row[2], int(row[3]), row[4], members, json.loads(row[5] or "{}"))


def _node_from_artifacts(db: sqlite3.Connection, node_id: str) -> RuntimeNode | None:
    # Prefer current non-stale artifact; fall back to immutable history so an older
    # compression run remains exportable after later rebuilds.
    row = db.execute(
        "SELECT payload_json FROM derived_artifacts WHERE node_id=? AND status!='stale' ORDER BY updated_at DESC LIMIT 1",
        (node_id,),
    ).fetchone()
    if row is None:
        row = db.execute(
            "SELECT payload_json FROM derived_artifact_versions WHERE node_id=? ORDER BY artifact_version_id DESC LIMIT 1",
            (node_id,),
        ).fetchone()
    if not row:
        return None
    payload = json.loads(row[0] or "{}")
    n = payload.get("node") or {}
    if n.get("node_id") != node_id:
        return None
    return RuntimeNode(
        n["node_id"], n["modality"], n["kind"], int(n["level"]), n.get("surface"),
        list(n.get("members") or []), dict(n.get("metadata") or {}),
    )


def _node_any(db: sqlite3.Connection, node_id: str) -> RuntimeNode:
    n = _node_from_canonical(db, node_id) or _node_from_artifacts(db, node_id)
    if n is None:
        raise KeyError(f"runtime pack references unknown node: {node_id}")
    return n


def build_runtime_view(db: sqlite3.Connection, compression_run_id: int, relation_policy: CompilePolicy | None = None) -> tuple[list[RuntimeNode], list[CompiledRelation]]:
    ensure_runtime_compression_schema(db)
    exists = db.execute(
        "SELECT 1 FROM runtime_compression_runs WHERE compression_run_id=?", (compression_run_id,)
    ).fetchone()
    if not exists:
        raise KeyError(f"unknown compression_run_id={compression_run_id}")

    nodes: dict[str, RuntimeNode] = {}
    for row in db.execute("SELECT node_id FROM nodes ORDER BY level,node_id").fetchall():
        n = _node_from_canonical(db, row[0])
        if n:
            nodes[n.node_id] = n

    rewrites: dict[str, list[str]] = {}
    for parent_id, rewritten_json in db.execute(
        "SELECT parent_id,rewritten_members_json FROM runtime_parent_rewrites WHERE compression_run_id=? ORDER BY parent_id",
        (compression_run_id,),
    ).fetchall():
        rewrites[parent_id] = list(json.loads(rewritten_json or "[]"))
        if parent_id not in nodes:
            nodes[parent_id] = _node_any(db, parent_id)

    promoted_ids = [r[0] for r in db.execute(
        "SELECT macro_node_id FROM runtime_promoted_macros WHERE compression_run_id=? ORDER BY macro_node_id",
        (compression_run_id,),
    ).fetchall()]
    for node_id in promoted_ids:
        if node_id not in nodes:
            nodes[node_id] = _node_any(db, node_id)

    # Resolve closure of every runtime member; derived parent/macro chains can reference
    # nodes that are not canonical inputs.
    pending = list(nodes)
    seen = set()
    while pending:
        node_id = pending.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        node = nodes[node_id]
        members = rewrites.get(node_id, node.members)
        for child in members:
            if child not in nodes:
                nodes[child] = _node_any(db, child)
                pending.append(child)

    runtime_nodes: list[RuntimeNode] = []
    for node_id, node in nodes.items():
        members = rewrites.get(node_id, node.members)
        runtime_nodes.append(RuntimeNode(
            node.node_id, node.modality, node.kind, node.level, node.surface,
            list(members), dict(node.metadata),
        ))
    runtime_nodes.sort(key=lambda n: (n.level, n.node_id))

    node_ids = set(nodes)
    relations = [r for r in aggregate_relations(db, relation_policy or CompilePolicy()) if r.from_id in node_ids and r.to_id in node_ids]
    relations.sort(key=lambda r: (r.from_id, r.to_id, r.relation_type))
    return runtime_nodes, relations


def _collect_strings(nodes: list[RuntimeNode], relations: list[CompiledRelation], pack_id: str, include_metadata: bool) -> list[str]:
    values = {pack_id}
    for n in nodes:
        values.update((n.node_id, n.modality, n.kind))
        if n.surface is not None:
            values.add(n.surface)
        if include_metadata and n.metadata:
            values.add(_stable_json(n.metadata))
    for r in relations:
        values.add(r.relation_type)
    return sorted(values)


def _encode_string_table(strings: list[str]) -> bytes:
    out = bytearray()
    for s in strings:
        raw = s.encode("utf-8")
        out.extend(_uvarint(len(raw)))
        out.extend(raw)
    return bytes(out)


def _decode_string_table(data: bytes, pos: int, count: int, end: int) -> tuple[list[str], int]:
    strings: list[str] = []
    for _ in range(count):
        size, pos = _read_uvarint(data, pos)
        if pos + size > end:
            raise PackFormatError("string table exceeds section boundary")
        try:
            strings.append(data[pos:pos+size].decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise PackFormatError("invalid UTF-8 in string table") from exc
        pos += size
    if pos != end:
        raise PackFormatError("string section contains trailing bytes")
    return strings, pos


def _encode_nodes(nodes: list[RuntimeNode], sid: dict[str, int], include_metadata: bool) -> tuple[bytes, list[int]]:
    index = {n.node_id: i for i, n in enumerate(nodes)}
    out = bytearray()
    offsets: list[int] = []
    for n in nodes:
        offsets.append(len(out))
        out.extend(_uvarint(sid[n.node_id]))
        out.extend(_uvarint(sid[n.modality]))
        out.extend(_uvarint(sid[n.kind]))
        out.extend(_uvarint(n.level))
        out.extend(_uvarint(0 if n.surface is None else sid[n.surface] + 1))
        metadata = _stable_json(n.metadata) if include_metadata and n.metadata else None
        out.extend(_uvarint(0 if metadata is None else sid[metadata] + 1))
        out.extend(_uvarint(len(n.members)))
        for member in n.members:
            if member not in index:
                raise KeyError(f"node {n.node_id} references member absent from runtime pack: {member}")
            out.extend(_uvarint(index[member]))
    return bytes(out), offsets


def _decode_nodes(data: bytes, start: int, end: int, count: int, strings: list[str], offsets: list[int]) -> list[RuntimeNode]:
    pos = start
    rows: list[tuple[str, str, str, int, str | None, dict[str, Any], list[int]]] = []
    for i in range(count):
        expected = start + offsets[i]
        if pos != expected:
            raise PackFormatError(f"node offset mismatch for record {i}: {pos} != {expected}")
        node_sid, pos = _read_uvarint(data, pos)
        modality_sid, pos = _read_uvarint(data, pos)
        kind_sid, pos = _read_uvarint(data, pos)
        level, pos = _read_uvarint(data, pos)
        surface_ref, pos = _read_uvarint(data, pos)
        metadata_ref, pos = _read_uvarint(data, pos)
        member_count, pos = _read_uvarint(data, pos)
        mids: list[int] = []
        for _ in range(member_count):
            mi, pos = _read_uvarint(data, pos)
            if mi >= count:
                raise PackFormatError(f"member index out of range: {mi}")
            mids.append(mi)
        try:
            node_id = strings[node_sid]
            modality = strings[modality_sid]
            kind = strings[kind_sid]
            surface = None if surface_ref == 0 else strings[surface_ref - 1]
            metadata = {} if metadata_ref == 0 else json.loads(strings[metadata_ref - 1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise PackFormatError("invalid node string reference or metadata") from exc
        rows.append((node_id, modality, kind, level, surface, metadata, mids))
    if pos != end:
        raise PackFormatError("node section size mismatch")
    ids = [r[0] for r in rows]
    return [RuntimeNode(r[0], r[1], r[2], r[3], r[4], [ids[i] for i in r[6]], r[5]) for r in rows]


def _encode_relations(relations: list[CompiledRelation], node_index: dict[str, int], sid: dict[str, int]) -> bytes:
    out = bytearray()
    for r in relations:
        out.extend(_uvarint(node_index[r.from_id]))
        out.extend(_uvarint(node_index[r.to_id]))
        out.extend(_uvarint(sid[r.relation_type]))
        out.extend(struct.pack("<ff", float(r.score), float(r.confidence)))
        out.extend(_uvarint(int(r.source_count)))
        out.extend(_uvarint(int(r.evidence_count)))
    return bytes(out)


def _decode_relations(data: bytes, start: int, end: int, count: int, nodes: list[RuntimeNode], strings: list[str]) -> list[CompiledRelation]:
    pos = start
    out: list[CompiledRelation] = []
    for _ in range(count):
        a, pos = _read_uvarint(data, pos)
        b, pos = _read_uvarint(data, pos)
        t, pos = _read_uvarint(data, pos)
        if pos + 8 > end:
            raise PackFormatError("truncated relation float payload")
        score, confidence = struct.unpack_from("<ff", data, pos)
        pos += 8
        sc, pos = _read_uvarint(data, pos)
        ec, pos = _read_uvarint(data, pos)
        if a >= len(nodes) or b >= len(nodes) or t >= len(strings):
            raise PackFormatError("relation reference out of range")
        out.append(CompiledRelation(nodes[a].node_id, nodes[b].node_id, strings[t], float(score), float(confidence), sc, ec))
    if pos != end:
        raise PackFormatError("relation section size mismatch")
    return out


def _encode_index(offsets: list[int]) -> bytes:
    # Fixed 32-bit offsets are compact enough for mobile packs and keep O(1) record seeks.
    if offsets and max(offsets) > 0xFFFFFFFF:
        raise ValueError("node section exceeds 4 GiB offset index limit")
    return b"".join(struct.pack("<I", x) for x in offsets)


def _decode_index(data: bytes, start: int, count: int, file_size: int) -> list[int]:
    expected = count * 4
    if start + expected != file_size:
        raise PackFormatError("index section/file size mismatch")
    return list(struct.unpack_from(f"<{count}I", data, start)) if count else []


def compile_base_pack(
    db: sqlite3.Connection,
    out_path: str | Path,
    *,
    pack_id: str = "equinox-base",
    pack_version: int = 1,
    compression_run_id: int | None = None,
    include_metadata: bool = True,
    persist: bool = True,
    verify_roundtrip: bool = True,
    relation_policy: CompilePolicy | None = None,
) -> RuntimePackManifest:
    t0 = time.perf_counter()
    ensure_runtime_pack_schema(db)
    run_id = int(compression_run_id or _latest_compression_run(db))
    nodes, relations = build_runtime_view(db, run_id, relation_policy=relation_policy)
    strings = _collect_strings(nodes, relations, pack_id, include_metadata)
    sid = {s: i for i, s in enumerate(strings)}
    node_index = {n.node_id: i for i, n in enumerate(nodes)}

    strings_blob = _encode_string_table(strings)
    nodes_blob, node_offsets = _encode_nodes(nodes, sid, include_metadata)
    relations_blob = _encode_relations(relations, node_index, sid)
    index_blob = _encode_index(node_offsets)

    strings_off = HEADER.size
    nodes_off = strings_off + len(strings_blob)
    relations_off = nodes_off + len(nodes_blob)
    index_off = relations_off + len(relations_blob)
    file_size = index_off + len(index_blob)
    payload = strings_blob + nodes_blob + relations_blob + index_blob
    payload_hash = hashlib.sha256(payload).digest()
    header = HEADER.pack(
        MAGIC, BINARY_MAJOR, BINARY_MINOR, HEADER.size, int(pack_version), run_id,
        len(nodes), len(relations), len(strings), sid[pack_id],
        strings_off, nodes_off, relations_off, index_off, file_size, payload_hash,
    )
    raw = header + payload
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(raw)
    full_hash = hashlib.sha256(raw).hexdigest()
    compile_ms = (time.perf_counter() - t0) * 1000.0

    verify_ms = 0.0
    valid = False
    if verify_roundtrip:
        tv = time.perf_counter()
        loaded = load_base_pack(out, verify_hash=True)
        source_node_sig = [
            (n.node_id, n.modality, n.kind, n.level, n.surface, tuple(n.members), n.metadata if include_metadata else {})
            for n in nodes
        ]
        loaded_node_sig = [
            (n.node_id, n.modality, n.kind, n.level, n.surface, tuple(n.members), n.metadata)
            for n in loaded.nodes
        ]
        # float32 is the intentional runtime representation for relation weights.
        def rel_sig(rs: list[CompiledRelation]):
            return [(r.from_id, r.to_id, r.relation_type, struct.pack("<f", float(r.score)), struct.pack("<f", float(r.confidence)), r.source_count, r.evidence_count) for r in rs]
        valid = source_node_sig == loaded_node_sig and rel_sig(relations) == rel_sig(loaded.relations)
        verify_ms = (time.perf_counter() - tv) * 1000.0
        if not valid:
            raise RuntimeError("runtime BASE round-trip mismatch")

    manifest = RuntimePackManifest(
        PACK_FORMAT, PACK_VERSION, BINARY_MAJOR, BINARY_MINOR, pack_id, int(pack_version), run_id,
        len(nodes), len(relations), len(strings), len(raw), HEADER.size,
        len(strings_blob), len(nodes_blob), len(relations_blob), len(index_blob),
        payload_hash.hex(), full_hash,
        (len(raw) / len(nodes)) if nodes else 0.0,
        (len(relations_blob) / len(relations)) if relations else 0.0,
        compile_ms, verify_ms, valid if verify_roundtrip else False, include_metadata,
    )
    if persist:
        db.execute(
            "INSERT INTO runtime_base_pack_runs(compression_run_id,pack_id,pack_version,output_path,file_bytes,node_count,relation_count,string_count,payload_sha256,full_file_sha256,roundtrip_valid,manifest_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, pack_id, int(pack_version), str(out), len(raw), len(nodes), len(relations), len(strings), payload_hash.hex(), full_hash, 1 if manifest.roundtrip_valid else 0, _stable_json(manifest.to_dict())),
        )
        db.commit()
    return manifest


def load_base_pack(path: str | Path, *, verify_hash: bool = True) -> LoadedRuntimePack:
    raw = Path(path).read_bytes()
    if len(raw) < HEADER.size:
        raise PackFormatError("file shorter than runtime BASE header")
    values = HEADER.unpack_from(raw, 0)
    (
        magic, major, minor, header_size, pack_version, compression_run_id,
        node_count, relation_count, string_count, pack_id_sid,
        strings_off, nodes_off, relations_off, index_off, file_size, payload_hash,
    ) = values
    if magic != MAGIC:
        raise PackFormatError("invalid runtime BASE magic")
    if major != BINARY_MAJOR:
        raise PackFormatError(f"unsupported runtime BASE major version {major}")
    if header_size != HEADER.size:
        raise PackFormatError(f"unexpected header size {header_size}")
    if file_size != len(raw):
        raise PackFormatError(f"declared file size {file_size} != actual {len(raw)}")
    if not (HEADER.size == strings_off <= nodes_off <= relations_off <= index_off <= file_size):
        raise PackFormatError("invalid section offsets")
    payload = raw[HEADER.size:]
    if verify_hash and hashlib.sha256(payload).digest() != payload_hash:
        raise PackFormatError("runtime BASE payload SHA-256 mismatch")

    strings, _ = _decode_string_table(raw, strings_off, string_count, nodes_off)
    if pack_id_sid >= len(strings):
        raise PackFormatError("pack_id string reference out of range")
    offsets = _decode_index(raw, index_off, node_count, file_size)
    if offsets and offsets != sorted(offsets):
        raise PackFormatError("node offsets are not monotonic")
    if offsets and offsets[0] != 0:
        raise PackFormatError("first node offset must be zero")
    if offsets and offsets[-1] >= relations_off - nodes_off:
        raise PackFormatError("node offset outside node section")
    nodes = _decode_nodes(raw, nodes_off, relations_off, node_count, strings, offsets)
    relations = _decode_relations(raw, relations_off, index_off, relation_count, nodes, strings)
    manifest = {
        "format": PACK_FORMAT,
        "binary_major": major,
        "binary_minor": minor,
        "pack_id": strings[pack_id_sid],
        "pack_version": pack_version,
        "compression_run_id": compression_run_id,
        "node_count": node_count,
        "relation_count": relation_count,
        "string_count": string_count,
        "file_bytes": file_size,
        "header_bytes": header_size,
        "string_table_bytes": nodes_off - strings_off,
        "node_section_bytes": relations_off - nodes_off,
        "relation_section_bytes": index_off - relations_off,
        "index_bytes": file_size - index_off,
        "payload_sha256": payload_hash.hex(),
        "full_file_sha256": hashlib.sha256(raw).hexdigest(),
    }
    return LoadedRuntimePack(manifest, nodes, relations, offsets)


def verify_base_pack(path: str | Path) -> dict[str, Any]:
    t0 = time.perf_counter()
    loaded = load_base_pack(path, verify_hash=True)
    ids = [n.node_id for n in loaded.nodes]
    unique_ids = len(ids) == len(set(ids))
    idset = set(ids)
    dangling_members = sorted({m for n in loaded.nodes for m in n.members if m not in idset})
    dangling_relations = sorted({x for r in loaded.relations for x in (r.from_id, r.to_id) if x not in idset})
    valid = unique_ids and not dangling_members and not dangling_relations
    return {
        **loaded.manifest,
        "valid": valid,
        "unique_node_ids": unique_ids,
        "dangling_members": dangling_members,
        "dangling_relation_endpoints": dangling_relations,
        "verify_ms": (time.perf_counter() - t0) * 1000.0,
    }
