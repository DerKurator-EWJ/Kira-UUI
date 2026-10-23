from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Iterable
import csv
import json
import math
import shutil
import time

from .behaviour_retention import (
    BehaviourRetentionCase,
    _direct_replay_policy,
    _summarize_cases,
    benchmark_teacher_probes_against_pack,
)
from .compiler import CompilePolicy
from .database import CanonicalDB
from .ir import ForgeDocument, IRNode, RelationEvidence
from .macro_discovery import MacroDiscoveryPolicy
from .runtime_compression import MacroPromotionPolicy, compile_runtime_compression
from .runtime_pack import compile_base_pack
from .runtime_cognition import RuntimeCognitionPolicy
from .teacher_behavior import (
    BehaviourProbeConfig,
    CausalTeacherBackend,
    TeacherProbe,
    extract_teacher_behaviour,
)
from .validate import validate_document
from .io import save_document

PARETO_FORMAT = "equinox-forge-behaviour-size-pareto-v1"
PARETO_VERSION = "0.18"


@dataclass(frozen=True)
class SweepPointConfig:
    evidence_top_k: int
    min_relation_score: float
    max_recall_nodes: int
    max_recall_depth: int
    compression_mode: str = "none"  # none | auto

    def key(self) -> str:
        score = f"{self.min_relation_score:.6f}".rstrip("0").rstrip(".") or "0"
        return (
            f"k{self.evidence_top_k}-s{score}-rn{self.max_recall_nodes}"
            f"-rd{self.max_recall_depth}-c{self.compression_mode}"
        ).replace(".", "p")


@dataclass
class ParetoSweepConfig:
    evidence_top_k_values: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    min_relation_score_values: list[float] = field(default_factory=lambda: [0.0, 0.02, 0.05])
    max_recall_nodes_values: list[int] = field(default_factory=lambda: [24, 96])
    max_recall_depth_values: list[int] = field(default_factory=lambda: [0, 3])
    compression_modes: list[str] = field(default_factory=lambda: ["none", "auto"])
    max_points: int = 256
    runtime_top_k: int | None = None
    include_pack_metadata: bool = False

    def validate(self, teacher_top_k: int) -> None:
        if teacher_top_k < 1:
            raise ValueError("teacher_top_k must be >= 1")
        if not self.evidence_top_k_values:
            raise ValueError("evidence_top_k_values cannot be empty")
        if any(x < 1 or x > teacher_top_k for x in self.evidence_top_k_values):
            raise ValueError("every evidence_top_k must be in [1, teacher_top_k]")
        if any(not (0.0 <= x <= 1.0) for x in self.min_relation_score_values):
            raise ValueError("min_relation_score values must be in [0,1]")
        if any(x < 1 for x in self.max_recall_nodes_values):
            raise ValueError("max_recall_nodes values must be >= 1")
        if any(x < 0 for x in self.max_recall_depth_values):
            raise ValueError("max_recall_depth values must be >= 0")
        if any(x not in {"none", "auto"} for x in self.compression_modes):
            raise ValueError("compression modes must be none or auto")
        if self.max_points < 1:
            raise ValueError("max_points must be >= 1")
        point_count = (
            len(set(self.evidence_top_k_values))
            * len(set(self.min_relation_score_values))
            * len(set(self.max_recall_nodes_values))
            * len(set(self.max_recall_depth_values))
            * len(set(self.compression_modes))
        )
        if point_count > self.max_points:
            raise ValueError(f"sweep has {point_count} points, above max_points={self.max_points}")

    def points(self) -> list[SweepPointConfig]:
        vals = product(
            sorted(set(self.evidence_top_k_values)),
            sorted(set(self.min_relation_score_values)),
            sorted(set(self.max_recall_nodes_values)),
            sorted(set(self.max_recall_depth_values)),
            sorted(set(self.compression_modes)),
        )
        return [SweepPointConfig(*x) for x in vals]


@dataclass
class ParetoPointResult:
    point_id: str
    config: dict[str, Any]
    pack_path: str
    pack_bytes: int
    pack_nodes: int
    pack_relations: int
    compression_promoted_macros: int
    compression_realized_ratio: float
    direct_top1_retention: float
    direct_teacher_topk_recall: float
    direct_teacher_mass_retained: float
    direct_ndcg: float
    direct_js: float
    full_top1_retention: float
    full_teacher_topk_recall: float
    full_teacher_mass_retained: float
    full_ndcg: float
    full_js: float
    average_recall_nodes: float
    average_relation_checks: float
    compile_elapsed_ms: float
    eval_elapsed_ms: float
    is_pareto_storage: bool = False
    is_pareto_runtime: bool = False
    is_pareto_joint: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParetoSweepReport:
    format: str
    version: str
    evaluation_scope: str
    generalization_claim: bool
    points: int
    teacher_top_k: int
    runtime_top_k: int
    extraction_elapsed_ms: float
    sweep_elapsed_ms: float
    teacher_probe_report: dict[str, Any]
    sweep_config: dict[str, Any]
    pareto_storage_ids: list[str]
    pareto_runtime_ids: list[str]
    pareto_joint_ids: list[str]
    smallest_pack_point: str | None
    highest_full_mass_point: str | None
    lowest_relation_work_point: str | None
    results: list[ParetoPointResult]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["results"] = [x.to_dict() for x in self.results]
        return d


def _filtered_document(doc: ForgeDocument, evidence_top_k: int, min_score: float) -> ForgeDocument:
    selected: list[RelationEvidence] = []
    for e in doc.evidence:
        rank = int((e.metadata or {}).get("rank", 1))
        if rank > evidence_top_k:
            continue
        if float(e.score) < float(min_score):
            continue
        selected.append(e)

    node_map = {n.node_id: n for n in doc.nodes}
    needed: set[str] = set()
    stack: list[str] = []
    for e in selected:
        stack.extend([e.from_id, e.to_id])
    while stack:
        nid = stack.pop()
        if nid in needed:
            continue
        node = node_map.get(nid)
        if node is None:
            continue
        needed.add(nid)
        stack.extend(node.members)
    nodes = [n for n in doc.nodes if n.node_id in needed]
    return ForgeDocument(format=doc.format, nodes=nodes, evidence=selected)


def _compression_policies(mode: str) -> tuple[MacroDiscoveryPolicy, MacroPromotionPolicy]:
    if mode == "auto":
        return (
            MacroDiscoveryPolicy(
                min_pattern_size=2,
                max_pattern_size=4,
                min_occurrences=2,
                min_parent_structures=2,
                definition_overhead=1.0,
                min_net_savings=1.0,
                min_compression_ratio=0.0,
                max_candidates=64,
                prune_dominated=True,
            ),
            MacroPromotionPolicy(
                min_realized_occurrences=2,
                min_realized_parent_structures=2,
                definition_overhead=1.0,
                min_realized_net_savings=1.0,
                min_realized_compression_ratio=0.0,
                max_promotions=32,
            ),
        )
    return (
        MacroDiscoveryPolicy(
            min_pattern_size=2,
            max_pattern_size=4,
            min_occurrences=1_000_000,
            min_parent_structures=1_000_000,
            definition_overhead=1.0,
            min_net_savings=1.0,
            min_compression_ratio=0.0,
            max_candidates=1,
            prune_dominated=True,
        ),
        MacroPromotionPolicy(
            min_realized_occurrences=1_000_000,
            min_realized_parent_structures=1_000_000,
            definition_overhead=1.0,
            min_realized_net_savings=1.0,
            min_realized_compression_ratio=0.0,
            max_promotions=1,
        ),
    )


def _dominates(a: ParetoPointResult, b: ParetoPointResult, mode: str) -> bool:
    eps = 1e-12
    if mode == "storage":
        # Lower bytes, higher compiler-only retention, lower direct divergence.
        comps = [
            (a.pack_bytes <= b.pack_bytes, a.pack_bytes < b.pack_bytes),
            (a.direct_top1_retention + eps >= b.direct_top1_retention, a.direct_top1_retention > b.direct_top1_retention + eps),
            (a.direct_teacher_mass_retained + eps >= b.direct_teacher_mass_retained, a.direct_teacher_mass_retained > b.direct_teacher_mass_retained + eps),
            (a.direct_ndcg + eps >= b.direct_ndcg, a.direct_ndcg > b.direct_ndcg + eps),
            (a.direct_js <= b.direct_js + eps, a.direct_js + eps < b.direct_js),
        ]
    elif mode == "runtime":
        comps = [
            (a.average_relation_checks <= b.average_relation_checks + eps, a.average_relation_checks + eps < b.average_relation_checks),
            (a.full_top1_retention + eps >= b.full_top1_retention, a.full_top1_retention > b.full_top1_retention + eps),
            (a.full_teacher_mass_retained + eps >= b.full_teacher_mass_retained, a.full_teacher_mass_retained > b.full_teacher_mass_retained + eps),
            (a.full_ndcg + eps >= b.full_ndcg, a.full_ndcg > b.full_ndcg + eps),
            (a.full_js <= b.full_js + eps, a.full_js + eps < b.full_js),
        ]
    else:
        comps = [
            (a.pack_bytes <= b.pack_bytes, a.pack_bytes < b.pack_bytes),
            (a.average_relation_checks <= b.average_relation_checks + eps, a.average_relation_checks + eps < b.average_relation_checks),
            (a.full_top1_retention + eps >= b.full_top1_retention, a.full_top1_retention > b.full_top1_retention + eps),
            (a.full_teacher_mass_retained + eps >= b.full_teacher_mass_retained, a.full_teacher_mass_retained > b.full_teacher_mass_retained + eps),
            (a.full_ndcg + eps >= b.full_ndcg, a.full_ndcg > b.full_ndcg + eps),
            (a.full_js <= b.full_js + eps, a.full_js + eps < b.full_js),
        ]
    return all(x[0] for x in comps) and any(x[1] for x in comps)


def _mark_frontier(results: list[ParetoPointResult], mode: str) -> list[str]:
    ids: list[str] = []
    for b in results:
        dominated = any(a.point_id != b.point_id and _dominates(a, b, mode) for a in results)
        flag = not dominated
        if mode == "storage":
            b.is_pareto_storage = flag
        elif mode == "runtime":
            b.is_pareto_runtime = flag
        else:
            b.is_pareto_joint = flag
        if flag:
            ids.append(b.point_id)
    return ids


def run_behaviour_pareto_sweep(
    backend: CausalTeacherBackend,
    prompts: Iterable[str],
    out_dir: str | Path,
    source_id: str,
    source_version: str,
    *,
    probe_cfg: BehaviourProbeConfig | None = None,
    base_cognition_policy: RuntimeCognitionPolicy | None = None,
    sweep_cfg: ParetoSweepConfig | None = None,
    pack_id_prefix: str = "equinox-pareto",
    pack_version: int = 18,
) -> ParetoSweepReport:
    probe_cfg = probe_cfg or BehaviourProbeConfig(top_k=8)
    probe_cfg.validate()
    sweep_cfg = sweep_cfg or ParetoSweepConfig(
        evidence_top_k_values=[1, min(2, probe_cfg.top_k), probe_cfg.top_k],
        min_relation_score_values=[0.0, 0.05],
        max_recall_nodes_values=[24, 96],
        max_recall_depth_values=[0, 3],
        compression_modes=["none", "auto"],
    )
    sweep_cfg.evidence_top_k_values = sorted(set(sweep_cfg.evidence_top_k_values))
    sweep_cfg.validate(probe_cfg.top_k)
    runtime_top_k = int(sweep_cfg.runtime_top_k or probe_cfg.top_k)
    if runtime_top_k < 1:
        raise ValueError("runtime_top_k must be >= 1")
    base_policy = base_cognition_policy or RuntimeCognitionPolicy(max_predictions=max(runtime_top_k, 8))
    base_policy.validate()

    prompts = [str(x) for x in prompts if str(x).strip()]
    if not prompts:
        raise ValueError("at least one non-empty prompt is required")
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    points_dir = out / "points"
    points_dir.mkdir(parents=True, exist_ok=True)

    t_extract = time.perf_counter()
    full_doc, probe_report, probes = extract_teacher_behaviour(
        backend, prompts, source_id, source_version, probe_cfg
    )
    extraction_elapsed_ms = (time.perf_counter() - t_extract) * 1000.0
    errors = [x for x in validate_document(full_doc) if x.severity == "error"]
    if errors:
        raise ValueError("teacher behaviour IR failed validation: " + "; ".join(x.message for x in errors))
    save_document(full_doc, out / "teacher-behaviour-full.eqir.json")
    (out / "teacher-probe-report.json").write_text(
        json.dumps(probe_report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    results: list[ParetoPointResult] = []
    t_sweep = time.perf_counter()
    for idx, point in enumerate(sweep_cfg.points(), 1):
        point_id = f"p{idx:04d}-{point.key()}"
        pdir = points_dir / point_id
        pdir.mkdir(parents=True, exist_ok=True)
        db_path = pdir / "point.sqlite"
        pack_path = pdir / "runtime.eqbase"
        filtered = _filtered_document(full_doc, point.evidence_top_k, point.min_relation_score)
        save_document(filtered, pdir / "compiled-input.eqir.json")
        errors = [x for x in validate_document(filtered) if x.severity == "error"]
        if errors:
            raise ValueError(f"filtered IR failed validation for {point_id}: " + "; ".join(x.message for x in errors))

        tc = time.perf_counter()
        with CanonicalDB(db_path) as db:
            for n in sorted(filtered.nodes, key=lambda n: (n.level, n.node_id)):
                db.upsert_node(n)
            for e in filtered.evidence:
                db.add_evidence(e)
            db.commit()
            discovery, promotion = _compression_policies(point.compression_mode)
            compression = compile_runtime_compression(
                db.db, discovery, promotion, persist=True, materialize=True
            )
            manifest = compile_base_pack(
                db.db,
                pack_path,
                pack_id=f"{pack_id_prefix}-{point_id}",
                pack_version=pack_version,
                compression_run_id=compression.compression_run_id,
                include_metadata=sweep_cfg.include_pack_metadata,
                persist=True,
                verify_roundtrip=True,
                # Evidence was already rank/score-pruned before node closure. Keep all
                # surviving relations so this axis is not applied twice implicitly.
                relation_policy=CompilePolicy(min_score=0.0, min_confidence=0.0, min_sources=1),
            )
        compile_elapsed_ms = (time.perf_counter() - tc) * 1000.0

        values = asdict(base_policy)
        values.update({
            "max_recall_nodes": point.max_recall_nodes,
            "max_recall_depth": point.max_recall_depth,
            "max_predictions": max(runtime_top_k, int(values.get("max_predictions", 0))),
        })
        full_policy = RuntimeCognitionPolicy(**values)
        direct_policy = _direct_replay_policy(full_policy, runtime_top_k)
        direct_cases, direct_ms = benchmark_teacher_probes_against_pack(
            pack_path, probes, probe_cfg,
            cognition_policy=direct_policy,
            teacher_top_k=probe_cfg.top_k,
            runtime_top_k=runtime_top_k,
        )
        full_cases, full_ms = benchmark_teacher_probes_against_pack(
            pack_path, probes, probe_cfg,
            cognition_policy=full_policy,
            teacher_top_k=probe_cfg.top_k,
            runtime_top_k=runtime_top_k,
        )
        direct = _summarize_cases(direct_cases)
        full = _summarize_cases(full_cases)
        result = ParetoPointResult(
            point_id=point_id,
            config=asdict(point),
            pack_path=str(pack_path),
            pack_bytes=manifest.file_bytes,
            pack_nodes=manifest.node_count,
            pack_relations=manifest.relation_count,
            compression_promoted_macros=compression.promoted_count,
            compression_realized_ratio=compression.realized_compression_ratio,
            direct_top1_retention=direct["top1_retention"],
            direct_teacher_topk_recall=direct["mean_teacher_topk_recall"],
            direct_teacher_mass_retained=direct["mean_teacher_probability_mass_retained_in_runtime_topk"],
            direct_ndcg=direct["mean_ndcg_at_k"],
            direct_js=direct["mean_conditional_js_divergence"],
            full_top1_retention=full["top1_retention"],
            full_teacher_topk_recall=full["mean_teacher_topk_recall"],
            full_teacher_mass_retained=full["mean_teacher_probability_mass_retained_in_runtime_topk"],
            full_ndcg=full["mean_ndcg_at_k"],
            full_js=full["mean_conditional_js_divergence"],
            average_recall_nodes=full["average_recall_nodes"],
            average_relation_checks=full["average_relation_checks"],
            compile_elapsed_ms=compile_elapsed_ms,
            eval_elapsed_ms=direct_ms + full_ms,
        )
        results.append(result)
        (pdir / "point-report.json").write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    sweep_elapsed_ms = (time.perf_counter() - t_sweep) * 1000.0
    storage_ids = _mark_frontier(results, "storage")
    runtime_ids = _mark_frontier(results, "runtime")
    joint_ids = _mark_frontier(results, "joint")

    smallest = min(results, key=lambda x: (x.pack_bytes, -x.full_teacher_mass_retained, x.point_id)).point_id if results else None
    highest_mass = max(results, key=lambda x: (x.full_teacher_mass_retained, x.full_ndcg, -x.pack_bytes, x.point_id)).point_id if results else None
    lowest_work = min(results, key=lambda x: (x.average_relation_checks, -x.full_teacher_mass_retained, x.pack_bytes, x.point_id)).point_id if results else None

    report = ParetoSweepReport(
        format=PARETO_FORMAT,
        version=PARETO_VERSION,
        evaluation_scope="compiled_probe_replay_pareto_sweep",
        generalization_claim=False,
        points=len(results),
        teacher_top_k=probe_cfg.top_k,
        runtime_top_k=runtime_top_k,
        extraction_elapsed_ms=extraction_elapsed_ms,
        sweep_elapsed_ms=sweep_elapsed_ms,
        teacher_probe_report=probe_report.to_dict(),
        sweep_config=asdict(sweep_cfg),
        pareto_storage_ids=storage_ids,
        pareto_runtime_ids=runtime_ids,
        pareto_joint_ids=joint_ids,
        smallest_pack_point=smallest,
        highest_full_mass_point=highest_mass,
        lowest_relation_work_point=lowest_work,
        results=results,
        notes=[
            "The teacher is probed once at the maximum configured top-k; all sweep points reuse the same captured evidence.",
            "This is compiled-probe replay, not held-out language generalization.",
            "evidence_top_k and min_relation_score prune evidence before node closure, so pack-size changes include removal of now-unreachable candidate nodes.",
            "Direct replay isolates compiler/pack loss; full runtime replay additionally includes bounded recall/routing effects.",
            "Pareto labels are multi-objective non-dominance labels, not a claim that one configuration is universally best.",
            "Wall-clock timings are host diagnostics only; deterministic relation-check counts are the portable runtime-work proxy in this milestone.",
        ],
    )
    (out / "pareto-report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    csv_path = out / "pareto-points.csv"
    fieldnames = [
        "point_id","evidence_top_k","min_relation_score","max_recall_nodes","max_recall_depth","compression_mode",
        "pack_bytes","pack_nodes","pack_relations","compression_promoted_macros","compression_realized_ratio",
        "direct_top1_retention","direct_teacher_mass_retained","direct_ndcg","direct_js",
        "full_top1_retention","full_teacher_mass_retained","full_ndcg","full_js",
        "average_recall_nodes","average_relation_checks","is_pareto_storage","is_pareto_runtime","is_pareto_joint",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {
                "point_id": r.point_id, **r.config,
                "pack_bytes": r.pack_bytes, "pack_nodes": r.pack_nodes, "pack_relations": r.pack_relations,
                "compression_promoted_macros": r.compression_promoted_macros,
                "compression_realized_ratio": r.compression_realized_ratio,
                "direct_top1_retention": r.direct_top1_retention,
                "direct_teacher_mass_retained": r.direct_teacher_mass_retained,
                "direct_ndcg": r.direct_ndcg, "direct_js": r.direct_js,
                "full_top1_retention": r.full_top1_retention,
                "full_teacher_mass_retained": r.full_teacher_mass_retained,
                "full_ndcg": r.full_ndcg, "full_js": r.full_js,
                "average_recall_nodes": r.average_recall_nodes,
                "average_relation_checks": r.average_relation_checks,
                "is_pareto_storage": r.is_pareto_storage,
                "is_pareto_runtime": r.is_pareto_runtime,
                "is_pareto_joint": r.is_pareto_joint,
            }
            writer.writerow(row)
    return report
