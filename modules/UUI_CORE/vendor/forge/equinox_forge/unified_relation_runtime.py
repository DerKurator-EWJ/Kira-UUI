from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import argparse
import hashlib
import json
import sqlite3

from .canonical_id_space import canonical_relation_id
from .canonical_relation_facets import RelationFacet
from .relation_macro_runtime import (
    RelationMacroNumericRuntime,
    RelationMacroRuntimePack,
    RelationMacroValue,
    _build_demo as _build_v053_demo,
    load_relation_macro_runtime_pack,
)

UNIFIED_RELATION_RUNTIME_FORMAT = "equinox-unified-numeric-relation-runtime-v1"
UNIFIED_RELATION_RUNTIME_VERSION = "0.54"


@dataclass(frozen=True)
class UnifiedRelationRoute:
    query_relation_id: int
    route_kind: str  # scalar | relation_macro
    execution_relation_id: int
    descendant_facet_ids: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_relation_id": int(self.query_relation_id),
            "route_kind": str(self.route_kind),
            "execution_relation_id": int(self.execution_relation_id),
            "descendant_facet_ids": [int(x) for x in self.descendant_facet_ids],
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "UnifiedRelationRoute":
        kind = str(row["route_kind"])
        if kind not in {"scalar", "relation_macro"}:
            raise ValueError(f"unsupported unified relation route kind: {kind!r}")
        return cls(
            int(row["query_relation_id"]),
            kind,
            int(row["execution_relation_id"]),
            tuple(int(x) for x in row.get("descendant_facet_ids", [])),
        )


@dataclass
class UnifiedRelationRuntimePack:
    format: str
    version: str
    relation_macro_pack: RelationMacroRuntimePack
    routes: list[UnifiedRelationRoute]
    digest: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            "format": self.format,
            "version": self.version,
            "relation_macro_pack": self.relation_macro_pack.to_payload(include_digest=True),
            "routes": [r.to_dict() for r in sorted(self.routes, key=lambda x: x.query_relation_id)],
            "metadata": dict(self.metadata),
        }
        if include_digest:
            payload["digest"] = self.digest
        return payload


@dataclass(frozen=True)
class UnifiedRelationValue:
    concrete_relation_id: int
    object_id: int
    route_kind: str
    concept_macro_id: int | None
    origin_subject_id: int | None
    inheritance_depth: int | None
    score: float
    confidence: float
    concept_nodes_touched: int
    concept_baseline_nodes_touched: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UnifiedRelationResolution:
    status: str
    subject_id: int
    query_relation_id: int
    execution_relation_id: int | None
    route_kind: str
    values: tuple[UnifiedRelationValue, ...]
    candidate_relations_examined: int
    route_nodes_touched: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "subject_id": int(self.subject_id),
            "query_relation_id": int(self.query_relation_id),
            "execution_relation_id": self.execution_relation_id,
            "route_kind": self.route_kind,
            "values": [v.to_dict() for v in self.values],
            "candidate_relations_examined": int(self.candidate_relations_examined),
            "route_nodes_touched": int(self.route_nodes_touched),
        }


def _stable_digest(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _relation_macro_pack_from_payload(payload: dict[str, Any]) -> RelationMacroRuntimePack:
    # Reuse the public loader's semantics without requiring a temporary file by
    # validating the same outer digest here and delegating embedded pack parsing
    # through the v0.53 module's dataclasses.
    from .relation_macro_runtime import (
        RelationMacroRoute,
        _faceted_pack_from_payload,
    )

    digest = str(payload.get("digest") or "")
    body = dict(payload)
    body.pop("digest", None)
    if digest != _stable_digest(body):
        raise ValueError("embedded relation macro runtime pack digest mismatch")
    return RelationMacroRuntimePack(
        str(payload["format"]),
        str(payload["version"]),
        _faceted_pack_from_payload(dict(payload["faceted_pack"])),
        int(payload["relation_macro_run_id"]),
        str(payload["relation_macro_digest"]),
        [RelationMacroRoute.from_dict(x) for x in payload["routes"]],
        digest,
        dict(payload.get("metadata") or {}),
    )


def _family_to_macro_routes(pack: RelationMacroRuntimePack) -> dict[int, int]:
    """Infer family query aliases from numeric structure only.

    A family ID is routed to a relation macro iff one macro's transitive facet set
    is exactly the set of all facets registered beneath that family. No symbolic
    relation names participate in the decision.
    """
    facets_by_family: dict[int, set[int]] = {}
    for facet in pack.faceted_pack.facets:
        facets_by_family.setdefault(int(facet.family_predicate_id), set()).add(int(facet.facet_predicate_id))

    macros_by_leafset: dict[frozenset[int], list[tuple[int, int]]] = {}
    for route in pack.routes:
        leafset = frozenset(int(x) for x in route.descendant_facet_ids)
        if not leafset:
            continue
        macros_by_leafset.setdefault(leafset, []).append((int(route.level), int(route.macro_id)))

    out: dict[int, int] = {}
    for family_id, facet_ids in sorted(facets_by_family.items()):
        matches = sorted(macros_by_leafset.get(frozenset(facet_ids), []))
        if not matches:
            continue
        # Prefer the shallowest exact macro; ID is only a deterministic tie-breaker.
        best_level = matches[0][0]
        best = [mid for level, mid in matches if level == best_level]
        if len(best) != 1:
            # Ambiguous compiler routing is safer left unresolved than guessed.
            continue
        out[int(family_id)] = int(best[0])
    return out


def build_unified_relation_runtime_pack(
    output_path: str | Path,
    *,
    relation_macro_pack: RelationMacroRuntimePack,
    metadata: dict[str, Any] | None = None,
) -> UnifiedRelationRuntimePack:
    """Compile one ID→ID routing table for scalar, family, and macro relations.

    Runtime dispatch never checks relation names or semantic enums. Families are
    compiler aliases to exact relation macros discovered structurally. Ordinary
    predicates and facet IDs use scalar execution; relation-macro IDs use macro
    execution directly.
    """
    scalar_ids = {
        int(r.predicate_id)
        for r in relation_macro_pack.faceted_pack.macro_pack.base.relations
    }
    facet_ids = {int(f.facet_predicate_id) for f in relation_macro_pack.faceted_pack.facets}
    scalar_ids.update(facet_ids)

    routes: dict[int, UnifiedRelationRoute] = {
        rid: UnifiedRelationRoute(rid, "scalar", rid, ())
        for rid in sorted(scalar_ids)
    }

    macro_routes = {int(r.macro_id): r for r in relation_macro_pack.routes}
    for mid, route in sorted(macro_routes.items()):
        routes[mid] = UnifiedRelationRoute(
            mid, "relation_macro", mid, tuple(int(x) for x in route.descendant_facet_ids)
        )

    # Compiler-only use of facet-family membership. The runtime receives only an
    # opaque numeric alias to an executable macro ID.
    for family_id, macro_id in sorted(_family_to_macro_routes(relation_macro_pack).items()):
        route = macro_routes[macro_id]
        routes[family_id] = UnifiedRelationRoute(
            family_id,
            "relation_macro",
            macro_id,
            tuple(int(x) for x in route.descendant_facet_ids),
        )

    pack = UnifiedRelationRuntimePack(
        UNIFIED_RELATION_RUNTIME_FORMAT,
        UNIFIED_RELATION_RUNTIME_VERSION,
        relation_macro_pack,
        list(routes.values()),
        "",
        dict(metadata or {}),
    )
    pack.digest = _stable_digest(pack.to_payload(include_digest=False))
    Path(output_path).write_text(
        json.dumps(pack.to_payload(include_digest=True), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return pack


def load_unified_relation_runtime_pack(path: str | Path) -> UnifiedRelationRuntimePack:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != UNIFIED_RELATION_RUNTIME_FORMAT or payload.get("version") != UNIFIED_RELATION_RUNTIME_VERSION:
        raise ValueError("unsupported unified numeric relation runtime pack")
    digest = str(payload.get("digest") or "")
    body = dict(payload)
    body.pop("digest", None)
    if digest != _stable_digest(body):
        raise ValueError("unified numeric relation runtime pack digest mismatch")
    return UnifiedRelationRuntimePack(
        str(payload["format"]),
        str(payload["version"]),
        _relation_macro_pack_from_payload(dict(payload["relation_macro_pack"])),
        [UnifiedRelationRoute.from_dict(x) for x in payload["routes"]],
        digest,
        dict(payload.get("metadata") or {}),
    )


class UnifiedNumericRelationRuntime:
    """One numeric query API for scalar relations, families, and relation macros.

    The runtime has no semantic branch for PROPERTY, ACTION, or any future family.
    All behavior is encoded in the compiled route table. A query relation ID either
    executes as a scalar predicate or as a relation-macro predicate set.
    """

    def __init__(self, pack: UnifiedRelationRuntimePack):
        self.pack = pack
        self.relation_macro_runtime = RelationMacroNumericRuntime(pack.relation_macro_pack)
        self._routes = {int(r.query_relation_id): r for r in pack.routes}

    def resolve(self, subject_id: int, query_relation_id: int) -> UnifiedRelationResolution:
        sid, qid = int(subject_id), int(query_relation_id)
        route = self._routes.get(qid)
        if route is None:
            return UnifiedRelationResolution("unknown_relation", sid, qid, None, "unknown", (), 0, 0)

        if route.route_kind == "relation_macro":
            result = self.relation_macro_runtime.resolve_relation_macro(sid, route.execution_relation_id)
            values = tuple(
                UnifiedRelationValue(
                    int(v.facet_predicate_id), int(v.object_id), str(v.route_kind), v.concept_macro_id,
                    v.origin_subject_id, v.inheritance_depth, float(v.score), float(v.confidence),
                    int(v.concept_nodes_touched), int(v.concept_baseline_nodes_touched),
                )
                for v in result.values
            )
            return UnifiedRelationResolution(
                result.status,
                sid,
                qid,
                int(route.execution_relation_id),
                "relation_macro",
                values,
                int(result.candidate_facets_examined),
                int(result.route_macro_nodes),
            )

        scalar = self.relation_macro_runtime.faceted_runtime.resolve(sid, route.execution_relation_id)
        r = scalar.resolution
        if r.status != "resolved" or r.object_id is None:
            return UnifiedRelationResolution(
                r.status, sid, qid, int(route.execution_relation_id), "scalar", (), 1, 1
            )
        value = UnifiedRelationValue(
            int(route.execution_relation_id), int(r.object_id), str(scalar.route_kind), scalar.macro_id,
            r.origin_subject_id, r.inheritance_depth, float(r.score), float(r.confidence),
            int(scalar.nodes_touched), int(scalar.baseline_nodes_touched),
        )
        return UnifiedRelationResolution(
            "resolved", sid, qid, int(route.execution_relation_id), "scalar", (value,), 1, 1
        )

    def resolve_surface(self, surface: str, query_relation_id: int, *, modality: str = "text", locale: str = "") -> UnifiedRelationResolution:
        ids = self.relation_macro_runtime.faceted_runtime.resolve_surface(surface, modality=modality, locale=locale)
        if len(ids) != 1:
            return UnifiedRelationResolution(
                "unknown_or_ambiguous_surface", 0, int(query_relation_id), None, "surface", (), 0, 0
            )
        return self.resolve(ids[0], query_relation_id)


def run_unified_relation_runtime_demo(out_dir: str | Path) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    source_out = out / "source"
    source_out.mkdir(parents=True, exist_ok=True)
    summary, v053_pack = _build_v053_demo(source_out)
    pack_path = out / "unified-relation.eqnum.json"
    pack = build_unified_relation_runtime_pack(
        pack_path,
        relation_macro_pack=v053_pack,
        metadata={"trial": UNIFIED_RELATION_RUNTIME_VERSION},
    )
    runtime = UnifiedNumericRelationRuntime(load_unified_relation_runtime_pack(pack_path))

    # Recover deterministic IDs without depending on names inside runtime routing.
    tmp = sqlite3.connect(":memory:")
    try:
        property_id = canonical_relation_id(tmp, "property")
        action_id = canonical_relation_id(tmp, "action")
        is_a_id = canonical_relation_id(tmp, "is_a")
    finally:
        tmp.close()

    subjects = sorted({int(r.subject_id) for r in pack.relation_macro_pack.faceted_pack.macro_pack.base.relations})
    chosen = None
    for sid in subjects:
        p = runtime.resolve(sid, property_id)
        a = runtime.resolve(sid, action_id)
        if len(p.values) >= 3 and len(a.values) >= 3:
            chosen = sid
            break
    if chosen is None:
        raise AssertionError("could not find unified-runtime demo leaf subject")

    property_result = runtime.resolve(chosen, property_id)
    action_result = runtime.resolve(chosen, action_id)
    root_macro_id = int(summary["root_relation_macro_id"])
    root_result = runtime.resolve(chosen, root_macro_id)
    scalar_result = runtime.resolve(chosen, is_a_id)

    legacy = RelationMacroNumericRuntime(v053_pack)
    # Validation-only baseline: the new runtime path must not call this API.
    legacy_property = legacy.faceted_runtime.resolve_family(chosen, property_id)
    legacy_action = legacy.faceted_runtime.resolve_family(chosen, action_id)
    legacy_prop_objects = {int(v.object_id) for v in legacy_property.values}
    legacy_act_objects = {int(v.object_id) for v in legacy_action.values}
    prop_objects = {int(v.object_id) for v in property_result.values}
    act_objects = {int(v.object_id) for v in action_result.values}
    root_objects = {int(v.object_id) for v in root_result.values}

    property_route = next(r for r in pack.routes if int(r.query_relation_id) == property_id)
    action_route = next(r for r in pack.routes if int(r.query_relation_id) == action_id)
    root_route = next(r for r in pack.routes if int(r.query_relation_id) == root_macro_id)
    scalar_route = next(r for r in pack.routes if int(r.query_relation_id) == is_a_id)

    # Prove family resolution is encoded entirely in the compiled numeric table:
    # sabotage the legacy family API and rerun; unified resolution must be unchanged.
    def _forbidden_family_api(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("legacy resolve_family API was used by unified runtime")

    runtime.relation_macro_runtime.faceted_runtime.resolve_family = _forbidden_family_api  # type: ignore[method-assign]
    property_after_sabotage = runtime.resolve(chosen, property_id)
    action_after_sabotage = runtime.resolve(chosen, action_id)

    report = {
        "format": UNIFIED_RELATION_RUNTIME_FORMAT,
        "version": UNIFIED_RELATION_RUNTIME_VERSION,
        "subject_id": int(chosen),
        "route_count": len(pack.routes),
        "family_alias_route_count": sum(
            1 for r in pack.routes
            if r.route_kind == "relation_macro" and int(r.query_relation_id) != int(r.execution_relation_id)
        ),
        "direct_macro_route_count": sum(
            1 for r in pack.routes
            if r.route_kind == "relation_macro" and int(r.query_relation_id) == int(r.execution_relation_id)
        ),
        "scalar_route_count": sum(1 for r in pack.routes if r.route_kind == "scalar"),
        "property_route": property_route.to_dict(),
        "action_route": action_route.to_dict(),
        "root_macro_route": root_route.to_dict(),
        "scalar_route": scalar_route.to_dict(),
        "property_query": property_result.to_dict(),
        "action_query": action_result.to_dict(),
        "root_macro_query": root_result.to_dict(),
        "scalar_query": scalar_result.to_dict(),
        "pack_digest": pack.digest,
        "controls": {
            "property_family_query_is_compiled_macro_alias": property_route.route_kind == "relation_macro" and property_route.execution_relation_id != property_id,
            "action_family_query_is_compiled_macro_alias": action_route.route_kind == "relation_macro" and action_route.execution_relation_id != action_id,
            "property_semantics_match_legacy_family_view": prop_objects == legacy_prop_objects,
            "action_semantics_match_legacy_family_view": act_objects == legacy_act_objects,
            "root_macro_still_resolves_union": root_objects == (prop_objects | act_objects),
            "ordinary_scalar_relation_uses_same_unified_api": scalar_route.route_kind == "scalar" and scalar_result.status == "resolved" and len(scalar_result.values) == 1,
            "runtime_does_not_depend_on_legacy_family_api": (
                {v.object_id for v in property_after_sabotage.values} == prop_objects
                and {v.object_id for v in action_after_sabotage.values} == act_objects
            ),
            "runtime_route_table_contains_no_symbolic_relation_names": all(
                isinstance(r.query_relation_id, int) and isinstance(r.execution_relation_id, int) for r in pack.routes
            ),
        },
    }
    if not all(report["controls"].values()):
        raise AssertionError(report["controls"])
    (out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Equinox v0.54 unified numeric relation routing demo.")
    ap.add_argument("--out", default="out/v0_54_unified_relation_runtime")
    args = ap.parse_args(argv)
    report = run_unified_relation_runtime_demo(args.out)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
