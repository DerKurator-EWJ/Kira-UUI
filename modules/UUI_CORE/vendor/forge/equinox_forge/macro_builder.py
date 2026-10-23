from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import hashlib
import json
import sqlite3

from .artifact_store import current_output_nodes, record_document_outputs
from .canonical import composite_id
from .ir import ForgeDocument, IRNode, RelationEvidence, RelationType, SourceRef, NodeKind, Modality
from .rebuild import source_policy_hash

MACRO_BUILDER_FORMAT = "equinox-forge-macro-builder-v1"
MACRO_BUILDER_VERSION = "0.11"


@dataclass
class MacroCompilePolicy:
    input_build_targets: list[str] = field(default_factory=list)
    input_nodes: list[str] = field(default_factory=list)
    allowed_statuses: list[str] = field(default_factory=lambda: ["stable", "active"])
    min_members: int = 2
    ordered_members: bool = False
    modality: str | None = None
    kind: str = NodeKind.COMPOSITE.value
    surface: str | None = None

    def validate(self) -> None:
        if self.min_members < 1:
            raise ValueError("min_members must be >= 1")
        if not self.input_build_targets and not self.input_nodes:
            raise ValueError("macro requires input_build_targets or input_nodes")
        if not self.allowed_statuses:
            raise ValueError("allowed_statuses must not be empty")
        if self.kind not in {x.value for x in NodeKind}:
            raise ValueError(f"unknown node kind: {self.kind}")
        if self.modality is not None and self.modality not in {x.value for x in Modality}:
            raise ValueError(f"unknown modality: {self.modality}")


@dataclass
class MacroBuildReport:
    format: str
    version: str
    target_id: str
    built: bool
    output_node_id: str | None
    members: list[str]
    member_sources: list[dict[str, str]]
    level: int | None
    status: str
    source_policy_hash: str
    policy_hash: str
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _node(db: sqlite3.Connection, node_id: str) -> IRNode:
    row = db.execute(
        "SELECT node_id,modality,kind,level,surface,metadata_json FROM nodes WHERE node_id=?", (node_id,)
    ).fetchone()
    if row:
        members = [r[0] for r in db.execute(
            "SELECT child_id FROM node_members WHERE parent_id=? ORDER BY position", (node_id,)
        ).fetchall()]
        return IRNode(row[0], row[1], row[2], int(row[3]), row[4], members, json.loads(row[5] or "{}"))
    # v0.11 can build a hierarchy without forcing every intermediate compiler
    # output into the canonical input tables. The current artifact payload is
    # therefore a valid read source for derived nodes.
    try:
        art = db.execute(
            "SELECT payload_json FROM derived_artifacts WHERE node_id=? AND status!='stale' ORDER BY updated_at DESC LIMIT 1",
            (node_id,),
        ).fetchone()
    except sqlite3.OperationalError:
        art = None
    if art:
        payload = json.loads(art[0] or "{}")
        n = payload.get("node") or {}
        if n.get("node_id") == node_id:
            return IRNode(
                n["node_id"], n["modality"], n["kind"], int(n["level"]),
                n.get("surface"), list(n.get("members") or []), dict(n.get("metadata") or {}),
            )
    raise KeyError(f"unknown macro member node: {node_id}")


def _add_closure(db: sqlite3.Connection, nodes: dict[str, IRNode], node_id: str) -> None:
    if node_id in nodes:
        return
    n = _node(db, node_id)
    nodes[node_id] = n
    for child in n.members:
        _add_closure(db, nodes, child)


def macro_policy_hash(policy: MacroCompilePolicy) -> str:
    raw = json.dumps(asdict(policy), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def resolve_macro_members(db: sqlite3.Connection, policy: MacroCompilePolicy) -> tuple[list[str], list[dict[str, str]]]:
    policy.validate()
    found: list[tuple[str, str, str]] = []  # node,status,origin
    for node_id in policy.input_nodes:
        _node(db, node_id)
        found.append((node_id, "explicit", f"node:{node_id}"))
    for target in policy.input_build_targets:
        for node_id, status in current_output_nodes(db, target, policy.allowed_statuses):
            found.append((node_id, status, f"build_target:{target}"))

    # Deduplicate by canonical node while preserving deterministic origin metadata.
    by_node: dict[str, tuple[str, str]] = {}
    for node_id, status, origin in sorted(found, key=lambda x: (x[0], x[2], x[1])):
        by_node.setdefault(node_id, (status, origin))
    members = list(by_node)
    if not policy.ordered_members:
        members.sort()
    else:
        explicit_order = [x for x in policy.input_nodes if x in by_node]
        dynamic = sorted(x for x in members if x not in explicit_order)
        members = explicit_order + dynamic
    meta = [{"node_id": nid, "status": by_node[nid][0], "origin": by_node[nid][1]} for nid in members]
    return members, meta


def compile_macro(db: sqlite3.Connection, target_id: str, policy: MacroCompilePolicy) -> tuple[ForgeDocument, MacroBuildReport]:
    policy.validate()
    members, member_sources = resolve_macro_members(db, policy)
    ph = macro_policy_hash(policy)
    sph = source_policy_hash(db)
    if len(members) < policy.min_members:
        return ForgeDocument(), MacroBuildReport(
            MACRO_BUILDER_FORMAT, MACRO_BUILDER_VERSION, target_id, False, None,
            members, member_sources, None, "not_built", sph, ph,
            f"only {len(members)} eligible members; requires {policy.min_members}",
        )

    member_nodes = [_node(db, x) for x in members]
    modalities = {n.modality for n in member_nodes}
    modality = policy.modality or (next(iter(modalities)) if len(modalities) == 1 else Modality.CROSSMODAL.value)
    label = policy.surface or f"macro:{target_id}"
    node_id = composite_id(modality, policy.kind, members, label)
    level = max(n.level for n in member_nodes) + 1
    macro = IRNode(
        node_id, modality, policy.kind, level, label, members,
        {
            "derived": True,
            "builder": MACRO_BUILDER_FORMAT,
            "builder_version": MACRO_BUILDER_VERSION,
            "build_target_id": target_id,
            "input_build_targets": sorted(set(policy.input_build_targets)),
            "allowed_statuses": list(policy.allowed_statuses),
            "ordered_members": bool(policy.ordered_members),
            "policy_hash": ph,
            "member_sources": member_sources,
        },
    )
    nodes: dict[str, IRNode] = {}
    for member in members:
        _add_closure(db, nodes, member)
    nodes[node_id] = macro
    src = SourceRef("equinox-forge-macro-builder", MACRO_BUILDER_VERSION, "hierarchical_composition")
    evidence = [
        RelationEvidence(node_id, member, RelationType.COMPOSITION.value, 1.0, 1.0, src, {
            "derived": True, "build_target_id": target_id, "policy_hash": ph,
        })
        for member in members
    ]
    doc = ForgeDocument(nodes=sorted(nodes.values(), key=lambda n: (n.level, n.node_id)), evidence=evidence)
    report = MacroBuildReport(
        MACRO_BUILDER_FORMAT, MACRO_BUILDER_VERSION, target_id, True, node_id,
        members, member_sources, level, "active", sph, ph, None,
    )
    return doc, report


def rebuild_macro(
    db: sqlite3.Connection,
    target_id: str,
    policy: MacroCompilePolicy,
    *,
    record_store: bool = True,
) -> tuple[ForgeDocument, MacroBuildReport]:
    from .dependency import register_macro_build_target, refresh_dependency_index

    doc, report = compile_macro(db, target_id, policy)
    output_hash = hashlib.sha256(json.dumps(doc.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    register_macro_build_target(db, target_id, policy, last_output_hash=output_hash)
    if record_store:
        outputs = [report.output_node_id] if report.output_node_id else []
        record_document_outputs(
            db, target_id, "macro", doc, outputs,
            status_by_node={} if not outputs else {report.output_node_id: "active"},
            source_policy_hash=report.source_policy_hash,
            artifact_type="macro_node",
        )
    refresh_dependency_index(db)
    db.commit()
    return doc, report
