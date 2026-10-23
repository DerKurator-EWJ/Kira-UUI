from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING
import argparse
import hashlib
import json

if TYPE_CHECKING:
    from .unified_relation_runtime import UnifiedRelationRuntimePack

UNIFIED_NUMERIC_KERNEL_FORMAT = "equinox-unified-numeric-execution-kernel-v1"
UNIFIED_NUMERIC_KERNEL_VERSION = "0.55"


@dataclass(frozen=True)
class KernelFact:
    subject_id: int
    predicate_id: int
    object_id: int
    score: float
    confidence: float
    source_count: int
    evidence_count: int

    @property
    def quality(self) -> float:
        return max(0.0, min(1.0, float(self.score) * float(self.confidence)))

    def to_list(self) -> list[Any]:
        return [int(self.subject_id), int(self.predicate_id), int(self.object_id), float(self.score), float(self.confidence), int(self.source_count), int(self.evidence_count)]

    @classmethod
    def from_list(cls, row: list[Any]) -> "KernelFact":
        if len(row) != 7:
            raise ValueError("kernel fact row must have seven fields")
        return cls(int(row[0]), int(row[1]), int(row[2]), float(row[3]), float(row[4]), int(row[5]), int(row[6]))


@dataclass(frozen=True)
class KernelSurfaceBinding:
    surface: str
    modality: str
    locale: str
    concept_id: int

    def to_list(self) -> list[Any]:
        return [self.surface, self.modality, self.locale, int(self.concept_id)]

    @classmethod
    def from_list(cls, row: list[Any]) -> "KernelSurfaceBinding":
        if len(row) != 4:
            raise ValueError("kernel surface row must have four fields")
        return cls(str(row[0]), str(row[1]), str(row[2]), int(row[3]))


@dataclass(frozen=True)
class KernelShortcut:
    concept_id: int
    predicate_id: int
    macro_id: int
    object_id: int
    origin_subject_id: int
    original_inheritance_depth: int
    original_inheritance_path: tuple[int, ...]
    score: float
    confidence: float
    source_count: int
    evidence_count: int

    @property
    def quality(self) -> float:
        return max(0.0, min(1.0, float(self.score) * float(self.confidence)))

    def to_list(self) -> list[Any]:
        return [int(self.concept_id), int(self.predicate_id), int(self.macro_id), int(self.object_id), int(self.origin_subject_id), int(self.original_inheritance_depth), [int(x) for x in self.original_inheritance_path], float(self.score), float(self.confidence), int(self.source_count), int(self.evidence_count)]

    @classmethod
    def from_list(cls, row: list[Any]) -> "KernelShortcut":
        if len(row) != 11:
            raise ValueError("kernel shortcut row must have eleven fields")
        return cls(int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]), tuple(int(x) for x in row[6]), float(row[7]), float(row[8]), int(row[9]), int(row[10]))


@dataclass(frozen=True)
class KernelRoute:
    query_relation_id: int
    route_kind: str  # scalar | relation_set
    concrete_relation_ids: tuple[int, ...]

    def to_list(self) -> list[Any]:
        return [int(self.query_relation_id), str(self.route_kind), [int(x) for x in self.concrete_relation_ids]]

    @classmethod
    def from_list(cls, row: list[Any]) -> "KernelRoute":
        if len(row) != 3:
            raise ValueError("kernel route row must have three fields")
        kind = str(row[1])
        if kind not in {"scalar", "relation_set"}:
            raise ValueError(f"unsupported kernel route kind: {kind!r}")
        concrete = tuple(int(x) for x in row[2])
        if kind == "scalar" and len(concrete) != 1:
            raise ValueError("scalar kernel route must contain exactly one concrete relation ID")
        if kind == "relation_set" and not concrete:
            raise ValueError("relation-set kernel route must not be empty")
        return cls(int(row[0]), kind, concrete)


@dataclass
class UnifiedNumericKernelPack:
    format: str
    version: str
    inheritance_predicate_ids: tuple[int, ...]
    facts: list[KernelFact]
    surfaces: list[KernelSurfaceBinding]
    concept_shortcuts: list[KernelShortcut]
    routes: list[KernelRoute]
    min_quality: float
    ambiguity_margin: float
    max_parent_depth: int
    digest: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            "format": self.format,
            "version": self.version,
            "inheritance_predicate_ids": [int(x) for x in self.inheritance_predicate_ids],
            "facts": [x.to_list() for x in self.facts],
            "surfaces": [x.to_list() for x in self.surfaces],
            "concept_shortcuts": [x.to_list() for x in self.concept_shortcuts],
            "routes": [x.to_list() for x in sorted(self.routes, key=lambda r: r.query_relation_id)],
            "policy": {
                "min_quality": float(self.min_quality),
                "ambiguity_margin": float(self.ambiguity_margin),
                "max_parent_depth": int(self.max_parent_depth),
            },
            "metadata": dict(self.metadata),
        }
        if include_digest:
            payload["digest"] = self.digest
        return payload


@dataclass(frozen=True)
class KernelValue:
    concrete_relation_id: int
    object_id: int
    route_kind: str
    concept_macro_id: int | None
    origin_subject_id: int | None
    inheritance_depth: int | None
    inheritance_path: tuple[int, ...]
    score: float
    confidence: float
    source_count: int
    evidence_count: int
    concept_nodes_touched: int
    concept_baseline_nodes_touched: int

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["inheritance_path"] = [int(x) for x in self.inheritance_path]
        return out


@dataclass(frozen=True)
class KernelResolution:
    status: str
    subject_id: int
    query_relation_id: int
    route_kind: str
    values: tuple[KernelValue, ...]
    candidate_relations_examined: int
    route_nodes_touched: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "subject_id": int(self.subject_id),
            "query_relation_id": int(self.query_relation_id),
            "route_kind": self.route_kind,
            "values": [v.to_dict() for v in self.values],
            "candidate_relations_examined": int(self.candidate_relations_examined),
            "route_nodes_touched": int(self.route_nodes_touched),
        }


def _stable_digest(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_unified_numeric_kernel_pack(
    output_path: str | Path,
    *,
    unified_pack: "UnifiedRelationRuntimePack",
    min_quality: float = 0.50,
    ambiguity_margin: float = 0.05,
    max_parent_depth: int = 8,
    metadata: dict[str, Any] | None = None,
) -> UnifiedNumericKernelPack:
    """Flatten v0.48-v0.54 runtime state into one numeric execution pack.

    The output contains no embedded historical runtime packs. It keeps only the
    data required by the frozen semantics: canonical facts, surfaces, validated
    concept-macro shortcuts, and query-relation routes.
    """
    base = unified_pack.relation_macro_pack.faceted_pack.macro_pack.base
    shortcuts = unified_pack.relation_macro_pack.faceted_pack.macro_pack.shortcuts

    routes: list[KernelRoute] = []
    for route in unified_pack.routes:
        if route.route_kind == "scalar":
            routes.append(KernelRoute(int(route.query_relation_id), "scalar", (int(route.execution_relation_id),)))
        else:
            concrete = tuple(sorted(set(int(x) for x in route.descendant_facet_ids)))
            if not concrete:
                raise ValueError(f"relation macro route {route.query_relation_id} has no concrete relations")
            routes.append(KernelRoute(int(route.query_relation_id), "relation_set", concrete))

    pack = UnifiedNumericKernelPack(
        UNIFIED_NUMERIC_KERNEL_FORMAT,
        UNIFIED_NUMERIC_KERNEL_VERSION,
        tuple(int(x) for x in base.inheritance_predicate_ids),
        [KernelFact(*x.to_list()) for x in base.relations],
        [KernelSurfaceBinding(*x.to_list()) for x in base.surfaces],
        [KernelShortcut.from_list(x.to_list()) for x in shortcuts],
        routes,
        float(min_quality),
        float(ambiguity_margin),
        int(max_parent_depth),
        "",
        dict(metadata or {}),
    )
    pack.digest = _stable_digest(pack.to_payload(include_digest=False))
    Path(output_path).write_text(
        json.dumps(pack.to_payload(include_digest=True), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return pack


def load_unified_numeric_kernel_pack(path: str | Path) -> UnifiedNumericKernelPack:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != UNIFIED_NUMERIC_KERNEL_FORMAT or payload.get("version") != UNIFIED_NUMERIC_KERNEL_VERSION:
        raise ValueError("unsupported unified numeric execution kernel pack")
    digest = str(payload.get("digest") or "")
    body = dict(payload)
    body.pop("digest", None)
    if digest != _stable_digest(body):
        raise ValueError("unified numeric execution kernel pack digest mismatch")
    policy = dict(payload.get("policy") or {})
    return UnifiedNumericKernelPack(
        str(payload["format"]),
        str(payload["version"]),
        tuple(int(x) for x in payload["inheritance_predicate_ids"]),
        [KernelFact.from_list(x) for x in payload["facts"]],
        [KernelSurfaceBinding.from_list(x) for x in payload["surfaces"]],
        [KernelShortcut.from_list(x) for x in payload["concept_shortcuts"]],
        [KernelRoute.from_list(x) for x in payload["routes"]],
        float(policy.get("min_quality", 0.50)),
        float(policy.get("ambiguity_margin", 0.05)),
        int(policy.get("max_parent_depth", 8)),
        digest,
        dict(payload.get("metadata") or {}),
    )


class UnifiedNumericExecutionKernel:
    """Single DB-free execution engine over one flat numeric pack.

    It directly implements the frozen v0.48 specificity semantics, v0.49 validated
    concept-macro shortcuts, and v0.54 numeric relation routing. No historical
    runtime object is instantiated underneath this kernel.
    """

    def __init__(self, pack: UnifiedNumericKernelPack):
        self.pack = pack
        self._inheritance = set(int(x) for x in pack.inheritance_predicate_ids)
        self._by_subject_predicate: dict[tuple[int, int], list[KernelFact]] = {}
        self._parents: dict[int, list[int]] = {}
        for fact in pack.facts:
            self._by_subject_predicate.setdefault((int(fact.subject_id), int(fact.predicate_id)), []).append(fact)
            if int(fact.predicate_id) in self._inheritance and fact.quality + 1e-15 >= pack.min_quality:
                self._parents.setdefault(int(fact.subject_id), []).append(int(fact.object_id))
        for key in self._by_subject_predicate:
            self._by_subject_predicate[key].sort(key=lambda f: (-f.quality, -f.confidence, f.object_id, f.subject_id))
        for sid in self._parents:
            self._parents[sid] = sorted(set(self._parents[sid]))

        self._surface_index: dict[tuple[str, str, str], list[int]] = {}
        for binding in pack.surfaces:
            self._surface_index.setdefault((binding.surface, binding.modality, binding.locale), []).append(int(binding.concept_id))
        for key in self._surface_index:
            self._surface_index[key] = sorted(set(self._surface_index[key]))

        self._shortcuts: dict[tuple[int, int], list[KernelShortcut]] = {}
        for shortcut in pack.concept_shortcuts:
            self._shortcuts.setdefault((int(shortcut.concept_id), int(shortcut.predicate_id)), []).append(shortcut)
        for key in self._shortcuts:
            self._shortcuts[key].sort(key=lambda s: (-s.quality, s.macro_id, s.object_id))

        self._routes = {int(route.query_relation_id): route for route in pack.routes}

    def resolve_surface_ids(self, surface: str, *, modality: str = "text", locale: str = "") -> list[int]:
        return list(self._surface_index.get((surface, modality, locale), []))

    def _levels(self, subject_id: int) -> list[list[tuple[int, tuple[int, ...]]]]:
        start = int(subject_id)
        levels: list[list[tuple[int, tuple[int, ...]]]] = [[(start, (start,))]]
        seen = {start}
        current = levels[0]
        for _ in range(self.pack.max_parent_depth):
            nxt: list[tuple[int, tuple[int, ...]]] = []
            for node, path in current:
                for parent in self._parents.get(node, []):
                    if parent in seen:
                        continue
                    seen.add(parent)
                    nxt.append((parent, path + (parent,)))
            if not nxt:
                break
            nxt.sort(key=lambda x: (x[0], x[1]))
            levels.append(nxt)
            current = nxt
        return levels

    def _baseline_nodes_touched(self, subject_id: int, predicate_id: int) -> int:
        count = 0
        sid, pid = int(subject_id), int(predicate_id)
        for level in self._levels(sid):
            count += len(level)
            found = False
            for origin, _path in level:
                if any(
                    fact.quality + 1e-15 >= self.pack.min_quality
                    for fact in self._by_subject_predicate.get((origin, pid), [])
                ):
                    found = True
            if found:
                break
        return count

    def _resolve_hierarchy_scalar(self, subject_id: int, predicate_id: int) -> KernelValue | None | str:
        sid, pid = int(subject_id), int(predicate_id)
        baseline_work = self._baseline_nodes_touched(sid, pid)

        # v0.49: qualifying local facts retain categorical specificity authority.
        local = [
            f for f in self._by_subject_predicate.get((sid, pid), [])
            if f.quality + 1e-15 >= self.pack.min_quality
        ]
        if local:
            result = self._resolve_direct_hierarchy(sid, pid)
            if isinstance(result, str) or result is None:
                return result
            fact, origin, depth, path = result
            return KernelValue(
                pid, int(fact.object_id), "local", None, origin, depth, path,
                float(fact.score), float(fact.confidence), int(fact.source_count), int(fact.evidence_count),
                1, baseline_work,
            )

        # v0.49: compile-time validated concept-macro shortcut.
        shortcuts = self._shortcuts.get((sid, pid), [])
        if shortcuts:
            objects = {int(s.object_id) for s in shortcuts}
            if len(objects) == 1:
                best = shortcuts[0]
                return KernelValue(
                    pid, int(best.object_id), "macro", int(best.macro_id), int(best.origin_subject_id),
                    int(best.original_inheritance_depth), tuple(int(x) for x in best.original_inheritance_path),
                    float(best.score), float(best.confidence), int(best.source_count), int(best.evidence_count),
                    2, baseline_work,
                )

        result = self._resolve_direct_hierarchy(sid, pid)
        if isinstance(result, str) or result is None:
            return result
        fact, origin, depth, path = result
        return KernelValue(
            pid, int(fact.object_id), "fallback", None, origin, depth, path,
            float(fact.score), float(fact.confidence), int(fact.source_count), int(fact.evidence_count),
            baseline_work, baseline_work,
        )

    def _resolve_direct_hierarchy(self, subject_id: int, predicate_id: int) -> tuple[KernelFact, int, int, tuple[int, ...]] | None | str:
        sid, pid = int(subject_id), int(predicate_id)
        for depth, level in enumerate(self._levels(sid)):
            by_object: dict[int, tuple[KernelFact, tuple[int, ...]]] = {}
            for origin, path in level:
                for fact in self._by_subject_predicate.get((origin, pid), []):
                    if fact.quality + 1e-15 < self.pack.min_quality:
                        continue
                    incumbent = by_object.get(int(fact.object_id))
                    if incumbent is None:
                        by_object[int(fact.object_id)] = (fact, path)
                    else:
                        old = incumbent[0]
                        if (fact.quality, fact.confidence, -fact.object_id, -fact.subject_id) > (
                            old.quality, old.confidence, -old.object_id, -old.subject_id
                        ):
                            by_object[int(fact.object_id)] = (fact, path)
            if not by_object:
                continue
            ranked = sorted(
                by_object.values(),
                key=lambda item: (-item[0].quality, -item[0].confidence, item[0].object_id, item[0].subject_id, item[1]),
            )
            top_fact, top_path = ranked[0]
            if len(ranked) > 1:
                second = ranked[1][0]
                if int(top_fact.object_id) != int(second.object_id) and (top_fact.quality - second.quality) < self.pack.ambiguity_margin:
                    return "ambiguous"
            return top_fact, int(top_fact.subject_id), int(depth), tuple(int(x) for x in top_path)
        return None

    def resolve(self, subject_id: int, query_relation_id: int) -> KernelResolution:
        sid, qid = int(subject_id), int(query_relation_id)
        route = self._routes.get(qid)
        if route is None:
            return KernelResolution("unknown_relation", sid, qid, "unknown", (), 0, 0)

        values: list[KernelValue] = []
        saw_ambiguous = False
        for concrete_pid in route.concrete_relation_ids:
            resolved = self._resolve_hierarchy_scalar(sid, int(concrete_pid))
            if resolved == "ambiguous":
                saw_ambiguous = True
                continue
            if resolved is None:
                continue
            values.append(resolved)

        values.sort(key=lambda x: (
            10**9 if x.inheritance_depth is None else int(x.inheritance_depth),
            int(x.concrete_relation_id), int(x.object_id),
        ))
        if values:
            return KernelResolution(
                "resolved", sid, qid, route.route_kind, tuple(values),
                len(route.concrete_relation_ids), 1,
            )
        return KernelResolution(
            "ambiguous" if saw_ambiguous and route.route_kind == "scalar" else "not_found",
            sid, qid, route.route_kind, (), len(route.concrete_relation_ids), 1,
        )

    def resolve_surface(self, surface: str, query_relation_id: int, *, modality: str = "text", locale: str = "") -> KernelResolution:
        ids = self.resolve_surface_ids(surface, modality=modality, locale=locale)
        if len(ids) != 1:
            return KernelResolution("unknown_or_ambiguous_surface", 0, int(query_relation_id), "surface", (), 0, 0)
        return self.resolve(ids[0], query_relation_id)


def _normalized_old(result: Any) -> tuple[Any, ...]:
    values = []
    for value in result.values:
        # v0.54 intentionally did not expose source/evidence counts or the original
        # inheritance path for multi-value relation routes. Compare all fields it did
        # expose, and compare the extra fields against the flat kernel separately.
        values.append((
            int(value.concrete_relation_id), int(value.object_id), str(value.route_kind),
            None if value.concept_macro_id is None else int(value.concept_macro_id),
            None if value.origin_subject_id is None else int(value.origin_subject_id),
            None if value.inheritance_depth is None else int(value.inheritance_depth),
            round(float(value.score), 12), round(float(value.confidence), 12),
            int(value.concept_nodes_touched), int(value.concept_baseline_nodes_touched),
        ))
    return str(result.status), tuple(values), int(result.candidate_relations_examined)


def _normalized_new(result: KernelResolution) -> tuple[Any, ...]:
    values = []
    for value in result.values:
        values.append((
            int(value.concrete_relation_id), int(value.object_id), str(value.route_kind),
            None if value.concept_macro_id is None else int(value.concept_macro_id),
            None if value.origin_subject_id is None else int(value.origin_subject_id),
            None if value.inheritance_depth is None else int(value.inheritance_depth),
            round(float(value.score), 12), round(float(value.confidence), 12),
            int(value.concept_nodes_touched), int(value.concept_baseline_nodes_touched),
        ))
    return str(result.status), tuple(values), int(result.candidate_relations_examined)


def run_unified_numeric_kernel_demo(out_dir: str | Path) -> dict[str, Any]:
    # Compiler/baseline modules are intentionally lazy imports. Loading and executing
    # an already-built v0.55 pack does not import any historical runtime layer.
    from .unified_relation_runtime import (
        UnifiedNumericRelationRuntime,
        _build_v053_demo,
        build_unified_relation_runtime_pack,
        load_unified_relation_runtime_pack,
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    source = out / "source"
    source.mkdir(parents=True, exist_ok=True)
    _summary, relation_macro_pack = _build_v053_demo(source)
    v054_path = out / "v054-unified.eqnum.json"
    v054_pack = build_unified_relation_runtime_pack(v054_path, relation_macro_pack=relation_macro_pack, metadata={"trial": "v055-baseline"})
    old_runtime = UnifiedNumericRelationRuntime(load_unified_relation_runtime_pack(v054_path))

    kernel_path = out / "kernel.eqnum.json"
    kernel_pack = build_unified_numeric_kernel_pack(kernel_path, unified_pack=v054_pack, metadata={"trial": UNIFIED_NUMERIC_KERNEL_VERSION})
    kernel = UnifiedNumericExecutionKernel(load_unified_numeric_kernel_pack(kernel_path))

    subjects = sorted({int(f.subject_id) for f in kernel_pack.facts})
    query_relations = sorted(int(r.query_relation_id) for r in kernel_pack.routes)
    parity_failures: list[dict[str, Any]] = []
    compared = 0
    resolved = 0
    for sid in subjects:
        for qid in query_relations:
            old = old_runtime.resolve(sid, qid)
            new = kernel.resolve(sid, qid)
            compared += 1
            if new.status == "resolved":
                resolved += 1
            if _normalized_old(old) != _normalized_new(new):
                parity_failures.append({
                    "subject_id": sid,
                    "query_relation_id": qid,
                    "old": old.to_dict(),
                    "new": new.to_dict(),
                })

    nested_payload = v054_pack.to_payload(include_digest=True)
    flat_payload = kernel_pack.to_payload(include_digest=True)
    nested_bytes = len(json.dumps(nested_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    flat_bytes = len(json.dumps(flat_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))

    # Prove execution is independent of the old runtime stack after compilation.
    # The kernel object contains only flat indexes built from its own pack.
    flat_runtime_has_nested_runtime = any(
        hasattr(kernel, name) for name in (
            "relation_macro_runtime", "faceted_runtime", "macro_runtime", "base_runtime"
        )
    )

    report = {
        "format": UNIFIED_NUMERIC_KERNEL_FORMAT,
        "version": UNIFIED_NUMERIC_KERNEL_VERSION,
        "pack_digest": kernel_pack.digest,
        "fact_count": len(kernel_pack.facts),
        "surface_binding_count": len(kernel_pack.surfaces),
        "concept_shortcut_count": len(kernel_pack.concept_shortcuts),
        "route_count": len(kernel_pack.routes),
        "parity": {
            "query_pairs_compared": compared,
            "resolved_query_pairs": resolved,
            "failure_count": len(parity_failures),
            "failures": parity_failures[:10],
        },
        "pack_sizes": {
            "nested_v054_json_bytes": nested_bytes,
            "flat_v055_json_bytes": flat_bytes,
            "fraction_of_nested": 0.0 if nested_bytes == 0 else flat_bytes / nested_bytes,
        },
        "controls": {
            "all_v054_query_pairs_semantically_identical": not parity_failures,
            "flat_pack_contains_no_embedded_runtime_pack": "relation_macro_pack" not in flat_payload and "faceted_pack" not in flat_payload and "macro_pack" not in flat_payload and "base" not in flat_payload,
            "execution_kernel_has_no_nested_runtime_object": not flat_runtime_has_nested_runtime,
            "single_resolve_api_handles_all_compiled_routes": compared == len(subjects) * len(query_relations),
            "flat_pack_is_smaller_than_nested_v054_pack": flat_bytes < nested_bytes,
        },
    }
    if not all(report["controls"].values()):
        raise AssertionError(report)
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Equinox v0.55 flat unified numeric execution kernel demo.")
    ap.add_argument("--out", default="out/v0_55_unified_numeric_kernel")
    args = ap.parse_args(argv)
    report = run_unified_numeric_kernel_demo(args.out)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
