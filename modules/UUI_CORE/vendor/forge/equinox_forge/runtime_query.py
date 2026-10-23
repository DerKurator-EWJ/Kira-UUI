from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable
import json
import math
import statistics
import time

from .ir import CompiledRelation
from .runtime_pack import LoadedRuntimePack, RuntimeNode, load_base_pack

RUNTIME_QUERY_FORMAT = "equinox-runtime-query-v1"
RUNTIME_QUERY_VERSION = "0.16"


@dataclass
class QueryMetrics:
    operation: str
    baseline_work_units: int
    indexed_work_units: int
    result_count: int
    reduction: float
    wall_ns_median: int | None = None
    baseline_wall_ns_median: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeIndexStats:
    node_count: int
    surface_keys: int
    parent_keys: int
    member_edges: int
    relation_from_keys: int
    relation_to_keys: int
    relation_count: int
    root_count: int
    build_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RoutingBenchmark:
    macro_node_id: str
    macro_member_count: int
    macro_leaf_count: int
    baseline_root_count: int
    baseline_matching_roots: int
    hierarchical_matching_roots: int
    matched_root_overlap: int
    coverage_of_baseline_matches: float
    baseline_symbol_comparisons: int
    hierarchical_index_lookups: int
    hierarchical_parent_edges: int
    hierarchical_work_units: int
    work_reduction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeBenchmarkReport:
    format: str
    version: str
    pack_manifest: dict[str, Any]
    index_stats: dict[str, Any]
    node_lookup: dict[str, Any]
    surface_lookup: dict[str, Any]
    relation_adjacency: dict[str, Any]
    parent_lookup: dict[str, Any]
    routing: list[dict[str, Any]]
    routing_summary: dict[str, Any]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _reduction(baseline: int | float, optimized: int | float) -> float:
    if baseline <= 0:
        return 0.0
    return max(-1.0, min(1.0, 1.0 - (float(optimized) / float(baseline))))


def _median_ns(fn: Callable[[], Any], repeat: int) -> int | None:
    if repeat <= 0:
        return None
    samples: list[int] = []
    # One warm-up keeps import/cache effects out of the tiny lookup timings.
    fn()
    for _ in range(repeat):
        t0 = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - t0)
    return int(statistics.median(samples)) if samples else None


class RuntimeIndex:
    """In-memory secondary indexes built only from the compiled .eqbase pack.

    This deliberately does not consult the Forge SQLite DB. The indexes therefore
    describe the same physical runtime graph that a device would load.
    """

    def __init__(self, pack: LoadedRuntimePack):
        t0 = time.perf_counter()
        self.pack = pack
        self.nodes = pack.nodes
        self.relations = pack.relations
        self.node_by_id: dict[str, RuntimeNode] = {}
        self.node_pos: dict[str, int] = {}
        self.surface_to_nodes: dict[str, list[str]] = {}
        self.parents: dict[str, list[str]] = {}
        self.children: dict[str, list[str]] = {}
        self.relations_from: dict[str, list[CompiledRelation]] = {}
        self.relations_to: dict[str, list[CompiledRelation]] = {}
        self.relation_by_key: dict[tuple[str, str, str], CompiledRelation] = {}

        for pos, node in enumerate(self.nodes):
            self.node_by_id[node.node_id] = node
            self.node_pos[node.node_id] = pos
            self.children[node.node_id] = list(node.members)
            if node.surface is not None:
                self.surface_to_nodes.setdefault(node.surface, []).append(node.node_id)
            for child in node.members:
                self.parents.setdefault(child, []).append(node.node_id)
        for key in self.surface_to_nodes:
            self.surface_to_nodes[key].sort()
        for key in self.parents:
            self.parents[key].sort()
        for rel in self.relations:
            self.relations_from.setdefault(rel.from_id, []).append(rel)
            self.relations_to.setdefault(rel.to_id, []).append(rel)
            self.relation_by_key[(rel.from_id, rel.to_id, rel.relation_type)] = rel
        for table in (self.relations_from, self.relations_to):
            for key in table:
                table[key].sort(key=lambda r: (r.relation_type, r.to_id, r.from_id))

        child_ids = set(self.parents)
        self.roots = sorted(n.node_id for n in self.nodes if n.members and n.node_id not in child_ids)
        self.root_set = set(self.roots)
        self._build_ms = (time.perf_counter() - t0) * 1000.0

    @classmethod
    def from_path(cls, path: str | Path, *, verify_hash: bool = True) -> "RuntimeIndex":
        return cls(load_base_pack(path, verify_hash=verify_hash))

    def stats(self) -> RuntimeIndexStats:
        return RuntimeIndexStats(
            node_count=len(self.nodes),
            surface_keys=len(self.surface_to_nodes),
            parent_keys=len(self.parents),
            member_edges=sum(len(n.members) for n in self.nodes),
            relation_from_keys=len(self.relations_from),
            relation_to_keys=len(self.relations_to),
            relation_count=len(self.relations),
            root_count=len(self.roots),
            build_ms=self._build_ms,
        )

    def lookup_node(self, node_id: str, *, timing_repeat: int = 0) -> tuple[RuntimeNode | None, QueryMetrics]:
        result = self.node_by_id.get(node_id)
        pos = self.node_pos.get(node_id)
        baseline = len(self.nodes) if pos is None else pos + 1
        indexed = 1
        linear_fn = lambda: next((n for n in self.nodes if n.node_id == node_id), None)
        index_fn = lambda: self.node_by_id.get(node_id)
        return result, QueryMetrics(
            "node_id_lookup", baseline, indexed, 1 if result else 0, _reduction(baseline, indexed),
            _median_ns(index_fn, timing_repeat), _median_ns(linear_fn, timing_repeat),
        )

    def lookup_surface(self, surface: str, *, timing_repeat: int = 0) -> tuple[list[RuntimeNode], QueryMetrics]:
        ids = self.surface_to_nodes.get(surface, [])
        results = [self.node_by_id[x] for x in ids]
        baseline = len(self.nodes)
        indexed = 1 + len(ids)
        linear_fn = lambda: [n for n in self.nodes if n.surface == surface]
        index_fn = lambda: [self.node_by_id[x] for x in self.surface_to_nodes.get(surface, [])]
        return results, QueryMetrics(
            "surface_lookup", baseline, indexed, len(results), _reduction(baseline, indexed),
            _median_ns(index_fn, timing_repeat), _median_ns(linear_fn, timing_repeat),
        )

    def parent_nodes(self, node_id: str, *, timing_repeat: int = 0) -> tuple[list[RuntimeNode], QueryMetrics]:
        ids = self.parents.get(node_id, [])
        results = [self.node_by_id[x] for x in ids]
        baseline = sum(len(n.members) for n in self.nodes)
        indexed = 1 + len(ids)
        linear_fn = lambda: [n for n in self.nodes if node_id in n.members]
        index_fn = lambda: [self.node_by_id[x] for x in self.parents.get(node_id, [])]
        return results, QueryMetrics(
            "parent_lookup", baseline, indexed, len(results), _reduction(baseline, indexed),
            _median_ns(index_fn, timing_repeat), _median_ns(linear_fn, timing_repeat),
        )

    def relation_neighbors(
        self,
        node_id: str,
        *,
        direction: str = "out",
        relation_type: str | None = None,
        timing_repeat: int = 0,
    ) -> tuple[list[CompiledRelation], QueryMetrics]:
        if direction not in {"out", "in"}:
            raise ValueError("direction must be 'out' or 'in'")
        table = self.relations_from if direction == "out" else self.relations_to
        bucket = table.get(node_id, [])
        results = [r for r in bucket if relation_type is None or r.relation_type == relation_type]
        baseline = len(self.relations)
        indexed = 1 + len(bucket)
        if direction == "out":
            linear_fn = lambda: [r for r in self.relations if r.from_id == node_id and (relation_type is None or r.relation_type == relation_type)]
        else:
            linear_fn = lambda: [r for r in self.relations if r.to_id == node_id and (relation_type is None or r.relation_type == relation_type)]
        index_fn = lambda: [r for r in table.get(node_id, []) if relation_type is None or r.relation_type == relation_type]
        return results, QueryMetrics(
            f"relation_{direction}_lookup", baseline, indexed, len(results), _reduction(baseline, indexed),
            _median_ns(index_fn, timing_repeat), _median_ns(linear_fn, timing_repeat),
        )

    def traverse_relation(
        self,
        start_id: str,
        relation_type: str,
        *,
        direction: str = "out",
        max_depth: int = 8,
    ) -> tuple[list[dict[str, Any]], QueryMetrics]:
        """Breadth-first traversal over one compiled relation channel.

        The method operates only on the loaded .eqbase indexes. It is intentionally
        generic so recursive concept IS_A chains and future relation hierarchies use
        the same runtime primitive. The start node itself is not returned.
        """
        if direction not in {"out", "in"}:
            raise ValueError("direction must be 'out' or 'in'")
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        table = self.relations_from if direction == "out" else self.relations_to
        queue: list[tuple[str, int]] = [(start_id, 0)]
        seen = {start_id}
        rows: list[dict[str, Any]] = []
        index_lookups = 0
        relation_checks = 0
        qpos = 0
        while qpos < len(queue):
            current, depth = queue[qpos]; qpos += 1
            if depth >= max_depth:
                continue
            index_lookups += 1
            bucket = table.get(current, [])
            relation_checks += len(bucket)
            for rel in bucket:
                if rel.relation_type != relation_type:
                    continue
                nxt = rel.to_id if direction == "out" else rel.from_id
                if nxt in seen:
                    continue
                seen.add(nxt)
                nd = depth + 1
                queue.append((nxt, nd))
                rows.append({
                    "node_id": nxt,
                    "depth": nd,
                    "score": float(rel.score),
                    "confidence": float(rel.confidence),
                    "source_count": int(rel.source_count),
                    "evidence_count": int(rel.evidence_count),
                })
        # A scan-based reference would inspect every compiled relation once per
        # frontier node. This is a deterministic work model, not a wall-clock claim.
        frontier_nodes = max(1, index_lookups)
        baseline = len(self.relations) * frontier_nodes
        indexed = index_lookups + relation_checks
        return rows, QueryMetrics(
            f"relation_traversal:{relation_type}", baseline, indexed, len(rows),
            _reduction(baseline, indexed), None, None,
        )

    def expand_leaves(self, node_id: str, *, max_visits: int = 1_000_000) -> tuple[list[str], dict[str, int]]:
        if node_id not in self.node_by_id:
            raise KeyError(node_id)
        leaves: list[str] = []
        stack: list[tuple[str, frozenset[str]]] = [(node_id, frozenset())]
        node_visits = 0
        edge_visits = 0
        while stack:
            current, path = stack.pop()
            node_visits += 1
            if node_visits > max_visits:
                raise RuntimeError(f"expansion exceeded max_visits={max_visits}")
            if current in path:
                raise RuntimeError(f"cycle detected while expanding {node_id}: {current}")
            node = self.node_by_id[current]
            if not node.members:
                leaves.append(current)
                continue
            edge_visits += len(node.members)
            new_path = path | {current}
            for child in reversed(node.members):
                stack.append((child, new_path))
        return leaves, {"node_visits": node_visits, "edge_visits": edge_visits, "leaf_count": len(leaves)}

    def ancestor_roots(self, node_id: str, *, max_visits: int = 1_000_000) -> tuple[list[str], dict[str, int]]:
        if node_id not in self.node_by_id:
            raise KeyError(node_id)
        queue = [node_id]
        seen: set[str] = set()
        roots: set[str] = set()
        lookups = 0
        parent_edges = 0
        while queue:
            current = queue.pop()
            if current in seen:
                continue
            seen.add(current)
            if len(seen) > max_visits:
                raise RuntimeError(f"ancestor traversal exceeded max_visits={max_visits}")
            ps = self.parents.get(current, [])
            lookups += 1
            parent_edges += len(ps)
            if not ps:
                if current != node_id and self.node_by_id[current].members:
                    roots.add(current)
                continue
            for parent in ps:
                if parent in self.root_set:
                    roots.add(parent)
                else:
                    queue.append(parent)
        return sorted(roots), {
            "index_lookups": lookups,
            "parent_edges": parent_edges,
            "work_units": lookups + parent_edges,
        }


def _naive_contains(haystack: list[str], needle: list[str]) -> tuple[bool, int]:
    if not needle:
        return True, 0
    if len(needle) > len(haystack):
        return False, 0
    comparisons = 0
    stop = len(haystack) - len(needle) + 1
    for i in range(stop):
        matched = True
        for j, expected in enumerate(needle):
            comparisons += 1
            if haystack[i + j] != expected:
                matched = False
                break
        if matched:
            return True, comparisons
    return False, comparisons


def _routing_benchmark(index: RuntimeIndex, macro_id: str, *, max_expanded_symbols: int = 2_000_000) -> RoutingBenchmark | None:
    macro = index.node_by_id.get(macro_id)
    if macro is None or len(macro.members) < 2:
        return None
    pattern, _ = index.expand_leaves(macro_id, max_visits=max_expanded_symbols)
    if len(pattern) < 2:
        return None
    roots = index.roots
    baseline_matches: set[str] = set()
    comparisons = 0
    expanded_total = 0
    for root_id in roots:
        expanded, _ = index.expand_leaves(root_id, max_visits=max_expanded_symbols)
        expanded_total += len(expanded)
        if expanded_total > max_expanded_symbols:
            return None
        found, c = _naive_contains(expanded, pattern)
        comparisons += c
        if found:
            baseline_matches.add(root_id)
    hierarchical_roots, m = index.ancestor_roots(macro_id, max_visits=max_expanded_symbols)
    hset = set(hierarchical_roots)
    overlap = len(hset & baseline_matches)
    coverage = overlap / len(baseline_matches) if baseline_matches else 1.0
    return RoutingBenchmark(
        macro_node_id=macro_id,
        macro_member_count=len(macro.members),
        macro_leaf_count=len(pattern),
        baseline_root_count=len(roots),
        baseline_matching_roots=len(baseline_matches),
        hierarchical_matching_roots=len(hset),
        matched_root_overlap=overlap,
        coverage_of_baseline_matches=coverage,
        baseline_symbol_comparisons=comparisons,
        hierarchical_index_lookups=m["index_lookups"],
        hierarchical_parent_edges=m["parent_edges"],
        hierarchical_work_units=m["work_units"],
        work_reduction=_reduction(comparisons, m["work_units"]),
    )


def _aggregate_metrics(metrics: list[QueryMetrics], operation: str) -> dict[str, Any]:
    if not metrics:
        return {
            "operation": operation,
            "samples": 0,
            "baseline_work_units": 0,
            "indexed_work_units": 0,
            "work_reduction": 0.0,
        }
    baseline = sum(x.baseline_work_units for x in metrics)
    indexed = sum(x.indexed_work_units for x in metrics)
    wall = [x.wall_ns_median for x in metrics if x.wall_ns_median is not None]
    bwall = [x.baseline_wall_ns_median for x in metrics if x.baseline_wall_ns_median is not None]
    out: dict[str, Any] = {
        "operation": operation,
        "samples": len(metrics),
        "baseline_work_units": baseline,
        "indexed_work_units": indexed,
        "work_reduction": _reduction(baseline, indexed),
        "result_items": sum(x.result_count for x in metrics),
    }
    if wall:
        out["indexed_wall_ns_median_of_medians"] = int(statistics.median(wall))
    if bwall:
        out["baseline_wall_ns_median_of_medians"] = int(statistics.median(bwall))
    return out


def benchmark_runtime_pack(
    pack: LoadedRuntimePack | str | Path,
    *,
    sample_nodes: int = 128,
    sample_macros: int = 16,
    timing_repeat: int = 50,
    max_expanded_symbols: int = 2_000_000,
) -> RuntimeBenchmarkReport:
    loaded = load_base_pack(pack) if isinstance(pack, (str, Path)) else pack
    index = RuntimeIndex(loaded)
    nodes = index.nodes
    if sample_nodes < 1:
        sample_nodes = 1
    # Deterministic even spread across the pack, not random sampling.
    if len(nodes) <= sample_nodes:
        sampled = nodes
    else:
        positions = sorted({round(i * (len(nodes) - 1) / (sample_nodes - 1)) for i in range(sample_nodes)}) if sample_nodes > 1 else [0]
        sampled = [nodes[i] for i in positions]

    node_metrics = [index.lookup_node(n.node_id, timing_repeat=timing_repeat)[1] for n in sampled]
    surfaced = [n for n in sampled if n.surface is not None]
    surface_metrics = [index.lookup_surface(n.surface or "", timing_repeat=timing_repeat)[1] for n in surfaced]
    parent_metrics = [index.parent_nodes(n.node_id, timing_repeat=timing_repeat)[1] for n in sampled]

    relation_nodes = [n for n in sampled if n.node_id in index.relations_from or n.node_id in index.relations_to]
    relation_metrics: list[QueryMetrics] = []
    for n in relation_nodes:
        relation_metrics.append(index.relation_neighbors(n.node_id, direction="out", timing_repeat=timing_repeat)[1])
        relation_metrics.append(index.relation_neighbors(n.node_id, direction="in", timing_repeat=timing_repeat)[1])

    macros = [n for n in nodes if len(n.members) >= 2 and n.node_id in index.parents]
    macros.sort(key=lambda n: (-len(index.parents.get(n.node_id, [])), -len(n.members), n.node_id))
    routing: list[RoutingBenchmark] = []
    for macro in macros[:max(0, sample_macros)]:
        try:
            r = _routing_benchmark(index, macro.node_id, max_expanded_symbols=max_expanded_symbols)
        except RuntimeError:
            r = None
        if r is not None:
            routing.append(r)

    baseline_route = sum(r.baseline_symbol_comparisons for r in routing)
    hierarchical_route = sum(r.hierarchical_work_units for r in routing)
    total_baseline_matches = sum(r.baseline_matching_roots for r in routing)
    total_overlap = sum(r.matched_root_overlap for r in routing)
    routing_summary = {
        "samples": len(routing),
        "baseline_symbol_comparisons": baseline_route,
        "hierarchical_work_units": hierarchical_route,
        "work_reduction": _reduction(baseline_route, hierarchical_route),
        "baseline_matching_roots": total_baseline_matches,
        "matched_root_overlap": total_overlap,
        "coverage_of_baseline_matches": (total_overlap / total_baseline_matches) if total_baseline_matches else 1.0,
    }

    return RuntimeBenchmarkReport(
        RUNTIME_QUERY_FORMAT,
        RUNTIME_QUERY_VERSION,
        dict(loaded.manifest),
        index.stats().to_dict(),
        _aggregate_metrics(node_metrics, "node_id_lookup"),
        _aggregate_metrics(surface_metrics, "surface_lookup"),
        _aggregate_metrics(relation_metrics, "relation_adjacency"),
        _aggregate_metrics(parent_metrics, "parent_lookup"),
        [x.to_dict() for x in routing],
        routing_summary,
        [
            "work_units are deterministic structural operations, not CPU instructions or energy measurements",
            "wall-clock nanoseconds are host-Python microbenchmarks and must not be treated as Android/runtime guarantees",
            "routing baseline naively scans fully expanded root symbol streams; hierarchical routing follows compiled parent indexes",
            "coverage can be below 1.0 when a pattern exists in roots that were intentionally not rewritten to the promoted macro",
        ],
    )


def query_runtime_pack(
    pack: str | Path,
    *,
    node_id: str | None = None,
    surface: str | None = None,
    parents_of: str | None = None,
    expand: str | None = None,
    relations_from: str | None = None,
    relations_to: str | None = None,
    relation_type: str | None = None,
    timing_repeat: int = 0,
) -> dict[str, Any]:
    index = RuntimeIndex.from_path(pack)
    result: dict[str, Any] = {
        "format": RUNTIME_QUERY_FORMAT,
        "version": RUNTIME_QUERY_VERSION,
        "pack": dict(index.pack.manifest),
        "index": index.stats().to_dict(),
    }
    if node_id is not None:
        node, metrics = index.lookup_node(node_id, timing_repeat=timing_repeat)
        result["query"] = {"type": "node", "node_id": node_id}
        result["result"] = None if node is None else asdict(node)
        result["metrics"] = metrics.to_dict()
    elif surface is not None:
        nodes, metrics = index.lookup_surface(surface, timing_repeat=timing_repeat)
        result["query"] = {"type": "surface", "surface": surface}
        result["result"] = [asdict(n) for n in nodes]
        result["metrics"] = metrics.to_dict()
    elif parents_of is not None:
        nodes, metrics = index.parent_nodes(parents_of, timing_repeat=timing_repeat)
        result["query"] = {"type": "parents", "node_id": parents_of}
        result["result"] = [asdict(n) for n in nodes]
        result["metrics"] = metrics.to_dict()
    elif expand is not None:
        leaves, metrics = index.expand_leaves(expand)
        result["query"] = {"type": "expand", "node_id": expand}
        result["result"] = leaves
        result["metrics"] = metrics
    elif relations_from is not None:
        rels, metrics = index.relation_neighbors(relations_from, direction="out", relation_type=relation_type, timing_repeat=timing_repeat)
        result["query"] = {"type": "relations_from", "node_id": relations_from, "relation_type": relation_type}
        result["result"] = [r.__dict__ for r in rels]
        result["metrics"] = metrics.to_dict()
    elif relations_to is not None:
        rels, metrics = index.relation_neighbors(relations_to, direction="in", relation_type=relation_type, timing_repeat=timing_repeat)
        result["query"] = {"type": "relations_to", "node_id": relations_to, "relation_type": relation_type}
        result["result"] = [r.__dict__ for r in rels]
        result["metrics"] = metrics.to_dict()
    else:
        raise ValueError("one query selector is required")
    return result
