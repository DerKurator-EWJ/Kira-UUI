from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import json
import math
import shutil
import time

from .database import CanonicalDB
from .compiler import CompilePolicy
from .macro_discovery import MacroDiscoveryPolicy
from .runtime_compression import MacroPromotionPolicy, compile_runtime_compression
from .runtime_pack import RuntimePackManifest, compile_base_pack, load_base_pack
from .runtime_cognition import RuntimeCognition, RuntimeCognitionPolicy
from .teacher_behavior import (
    BehaviourProbeConfig,
    BehaviourProbeReport,
    CausalTeacherBackend,
    TeacherCandidate,
    TeacherProbe,
    canonical_candidate_surface,
    extract_teacher_behaviour,
)
from .validate import validate_document
from .io import save_document

RETENTION_FORMAT = "equinox-forge-behaviour-retention-v1"
RETENTION_VERSION = "0.17"


@dataclass
class BehaviourRetentionCase:
    prompt: str
    teacher_top1: str | None
    runtime_top1: str | None
    top1_match: bool
    teacher_top1_runtime_rank: int | None
    teacher_candidates: list[dict[str, Any]]
    runtime_candidates: list[dict[str, Any]]
    overlap_count: int
    teacher_topk_recall: float
    runtime_precision_against_teacher_topk: float
    teacher_probability_mass_captured_by_probe: float
    teacher_probability_mass_retained_in_runtime_topk: float
    ndcg_at_k: float
    conditional_js_divergence: float
    conditional_total_variation: float
    resolved_context: bool
    recall_nodes: int
    relation_checks: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BehaviourRetentionReport:
    format: str
    version: str
    evaluation_scope: str
    generalization_claim: bool
    runtime_scores_are_calibrated_probabilities: bool
    cases: int
    teacher_top_k: int
    runtime_top_k: int
    top1_retention: float
    mean_teacher_topk_recall: float
    mean_runtime_precision_against_teacher_topk: float
    mean_teacher_probability_mass_captured_by_probe: float
    mean_teacher_probability_mass_retained_in_runtime_topk: float
    mean_ndcg_at_k: float
    mean_conditional_js_divergence: float
    mean_conditional_total_variation: float
    direct_replay: dict[str, Any]
    unresolved_contexts: int
    no_runtime_predictions: int
    average_recall_nodes: float
    average_relation_checks: float
    teacher_probe_elapsed_ms: float
    runtime_eval_elapsed_ms: float
    pack_bytes: int
    pack_nodes: int
    pack_relations: int
    compiled_prediction_evidence: int
    bytes_per_compiled_prompt: float
    bytes_per_retained_top1: float | None
    teacher_model_bytes: int | None
    pack_to_teacher_storage_ratio: float | None
    backend_manifest: dict[str, Any]
    probe_report: dict[str, Any]
    pack_manifest: dict[str, Any]
    relation_compile_policy: dict[str, Any]
    cases_detail: list[BehaviourRetentionCase]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cases_detail"] = [x.to_dict() for x in self.cases_detail]
        return d


def _aggregate_teacher_candidates(
    candidates: Iterable[TeacherCandidate],
    cfg: BehaviourProbeConfig,
    top_k: int,
) -> list[tuple[str, float]]:
    """Collapse teacher-token aliases onto canonical surface strings.

    Different tokenizer ids can decode to the same canonical surface. Behaviour
    retention is measured at the Equinox surface boundary, not teacher token-id
    identity, so their probability mass is summed before ranking.
    """
    scores: dict[str, float] = {}
    first_rank: dict[str, int] = {}
    for rank, cand in enumerate(candidates, 1):
        surface = canonical_candidate_surface(cand, cfg)
        if not surface:
            continue
        scores[surface] = scores.get(surface, 0.0) + max(0.0, float(cand.probability))
        first_rank.setdefault(surface, rank)
    ranked = sorted(scores.items(), key=lambda x: (-x[1], first_rank[x[0]], x[0]))
    return ranked[:top_k]


def _aggregate_runtime_candidates(report, top_k: int) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    first_rank: dict[str, int] = {}
    for rank, cand in enumerate(report.predictions, 1):
        if not cand.surface:
            continue
        scores[cand.surface] = scores.get(cand.surface, 0.0) + max(0.0, float(cand.score))
        first_rank.setdefault(cand.surface, rank)
    ranked = sorted(scores.items(), key=lambda x: (-x[1], first_rank[x[0]], x[0]))
    return ranked[:top_k]


def _normalize(weights: dict[str, float], support: list[str]) -> list[float]:
    vals = [max(0.0, float(weights.get(x, 0.0))) for x in support]
    total = sum(vals)
    if total <= 0.0:
        return [0.0 for _ in support]
    return [x / total for x in vals]


def _js_divergence_bits(p: list[float], q: list[float]) -> float:
    if len(p) != len(q):
        raise ValueError("distribution length mismatch")
    if not p:
        return 0.0
    if sum(p) <= 0.0 and sum(q) <= 0.0:
        return 0.0
    if sum(p) <= 0.0 or sum(q) <= 0.0:
        return 1.0
    m = [(a + b) * 0.5 for a, b in zip(p, q)]
    def kl(a: list[float], b: list[float]) -> float:
        out = 0.0
        for x, y in zip(a, b):
            if x > 0.0:
                out += x * math.log2(x / y)
        return out
    return max(0.0, min(1.0, 0.5 * kl(p, m) + 0.5 * kl(q, m)))


def _ndcg(runtime_surfaces: list[str], teacher_relevance: dict[str, float], k: int) -> float:
    if k <= 0:
        return 0.0
    def dcg(items: list[str]) -> float:
        total = 0.0
        for i, surface in enumerate(items[:k], 1):
            rel = max(0.0, teacher_relevance.get(surface, 0.0))
            total += rel / math.log2(i + 1.0)
        return total
    ideal = [x for x, _ in sorted(teacher_relevance.items(), key=lambda x: (-x[1], x[0]))[:k]]
    denom = dcg(ideal)
    return (dcg(runtime_surfaces) / denom) if denom > 0.0 else 0.0


def _directory_bytes(path: str | Path | None) -> int | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    if p.is_file():
        return p.stat().st_size
    total = 0
    try:
        for child in p.rglob("*"):
            if child.is_file():
                total += child.stat().st_size
    except OSError:
        return None
    return total


def benchmark_teacher_probes_against_pack(
    pack_path: str | Path,
    probes: list[TeacherProbe],
    probe_cfg: BehaviourProbeConfig,
    *,
    cognition_policy: RuntimeCognitionPolicy | None = None,
    teacher_top_k: int | None = None,
    runtime_top_k: int | None = None,
) -> tuple[list[BehaviourRetentionCase], float]:
    teacher_top_k = int(teacher_top_k or probe_cfg.top_k)
    runtime_top_k = int(runtime_top_k or teacher_top_k)
    if teacher_top_k < 1 or runtime_top_k < 1:
        raise ValueError("top-k values must be >= 1")
    policy = cognition_policy or RuntimeCognitionPolicy(max_predictions=max(runtime_top_k, 8))
    if policy.max_predictions < runtime_top_k:
        policy = RuntimeCognitionPolicy(**{**asdict(policy), "max_predictions": runtime_top_k})
    cog = RuntimeCognition(load_base_pack(pack_path), policy)

    start = time.perf_counter()
    rows: list[BehaviourRetentionCase] = []
    for probe in probes:
        teacher = _aggregate_teacher_candidates(probe.candidates, probe_cfg, teacher_top_k)
        runtime_report = cog.predict(probe.prompt)
        runtime = _aggregate_runtime_candidates(runtime_report, runtime_top_k)

        teacher_surfaces = [x for x, _ in teacher]
        runtime_surfaces = [x for x, _ in runtime]
        teacher_weights = dict(teacher)
        runtime_weights = dict(runtime)
        teacher_top1 = teacher_surfaces[0] if teacher_surfaces else None
        runtime_top1 = runtime_surfaces[0] if runtime_surfaces else None
        rank = None
        if teacher_top1 is not None:
            try:
                rank = runtime_surfaces.index(teacher_top1) + 1
            except ValueError:
                pass
        overlap = set(teacher_surfaces) & set(runtime_surfaces)
        teacher_mass = sum(teacher_weights.values())
        retained_mass = sum(teacher_weights.get(x, 0.0) for x in runtime_surfaces)

        support = sorted(set(teacher_surfaces) | set(runtime_surfaces))
        p = _normalize(teacher_weights, support)
        q = _normalize(runtime_weights, support)
        tv = 0.5 * sum(abs(a - b) for a, b in zip(p, q)) if support else 0.0

        relation_checks = runtime_report.work.get("relation_checks", 0) + runtime_report.work.get("prediction_relation_checks", 0)
        rows.append(BehaviourRetentionCase(
            prompt=probe.prompt,
            teacher_top1=teacher_top1,
            runtime_top1=runtime_top1,
            top1_match=(teacher_top1 is not None and teacher_top1 == runtime_top1),
            teacher_top1_runtime_rank=rank,
            teacher_candidates=[{"surface": s, "probability": p} for s, p in teacher],
            runtime_candidates=[{"surface": s, "score": v} for s, v in runtime],
            overlap_count=len(overlap),
            teacher_topk_recall=(len(overlap) / len(teacher_surfaces) if teacher_surfaces else 0.0),
            runtime_precision_against_teacher_topk=(len(overlap) / len(runtime_surfaces) if runtime_surfaces else 0.0),
            teacher_probability_mass_captured_by_probe=teacher_mass,
            teacher_probability_mass_retained_in_runtime_topk=(retained_mass / teacher_mass if teacher_mass > 0.0 else 0.0),
            ndcg_at_k=_ndcg(runtime_surfaces, teacher_weights, runtime_top_k),
            conditional_js_divergence=_js_divergence_bits(p, q),
            conditional_total_variation=tv,
            resolved_context=bool(runtime_report.resolved_context_ids),
            recall_nodes=runtime_report.work.get("recall_nodes", 0),
            relation_checks=relation_checks,
        ))
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return rows, elapsed_ms


def _summarize_cases(cases: list[BehaviourRetentionCase]) -> dict[str, Any]:
    n = len(cases)
    if not n:
        return {
            "cases": 0, "top1_retention": 0.0, "mean_teacher_topk_recall": 0.0,
            "mean_runtime_precision_against_teacher_topk": 0.0,
            "mean_teacher_probability_mass_captured_by_probe": 0.0,
            "mean_teacher_probability_mass_retained_in_runtime_topk": 0.0,
            "mean_ndcg_at_k": 0.0, "mean_conditional_js_divergence": 0.0,
            "mean_conditional_total_variation": 0.0, "unresolved_contexts": 0,
            "no_runtime_predictions": 0, "average_recall_nodes": 0.0,
            "average_relation_checks": 0.0,
        }
    def avg(name: str) -> float:
        return sum(float(getattr(x, name)) for x in cases) / n
    return {
        "cases": n,
        "top1_retention": sum(1 for x in cases if x.top1_match) / n,
        "mean_teacher_topk_recall": avg("teacher_topk_recall"),
        "mean_runtime_precision_against_teacher_topk": avg("runtime_precision_against_teacher_topk"),
        "mean_teacher_probability_mass_captured_by_probe": avg("teacher_probability_mass_captured_by_probe"),
        "mean_teacher_probability_mass_retained_in_runtime_topk": avg("teacher_probability_mass_retained_in_runtime_topk"),
        "mean_ndcg_at_k": avg("ndcg_at_k"),
        "mean_conditional_js_divergence": avg("conditional_js_divergence"),
        "mean_conditional_total_variation": avg("conditional_total_variation"),
        "unresolved_contexts": sum(1 for x in cases if not x.resolved_context),
        "no_runtime_predictions": sum(1 for x in cases if not x.runtime_candidates),
        "average_recall_nodes": avg("recall_nodes"),
        "average_relation_checks": avg("relation_checks"),
    }


def _direct_replay_policy(base: RuntimeCognitionPolicy, runtime_top_k: int) -> RuntimeCognitionPolicy:
    # Isolate compiler/pack retention from associative recall. Behaviour-only packs
    # contain direct context→prediction edges, so depth=0 yields a clean replay.
    values = asdict(base)
    values.update({
        "max_recall_depth": 0,
        "max_predictions": max(runtime_top_k, int(values.get("max_predictions", 0))),
        "min_prediction_score": 0.0,
    })
    return RuntimeCognitionPolicy(**values)


def compile_and_benchmark_teacher_behaviour(
    backend: CausalTeacherBackend,
    prompts: Iterable[str],
    out_dir: str | Path,
    source_id: str,
    source_version: str,
    *,
    probe_cfg: BehaviourProbeConfig | None = None,
    cognition_policy: RuntimeCognitionPolicy | None = None,
    runtime_top_k: int | None = None,
    pack_id: str = "equinox-behaviour-retention",
    pack_version: int = 17,
    include_pack_metadata: bool = False,
) -> BehaviourRetentionReport:
    """Compile controlled teacher behaviour into a BASE pack and replay it.

    This milestone deliberately measures *compiled probe replay retention*. The same
    controlled prompts used to obtain teacher behaviour are replayed against the
    compiled runtime. It tests whether the compiler/runtime preserves captured
    behaviour; it is not a held-out language generalization benchmark.
    """
    cfg = probe_cfg or BehaviourProbeConfig()
    cfg.validate()
    prompts = [str(x) for x in prompts if str(x).strip()]
    if not prompts:
        raise ValueError("at least one non-empty prompt is required")
    runtime_top_k = int(runtime_top_k or cfg.top_k)
    policy = cognition_policy or RuntimeCognitionPolicy(max_predictions=max(runtime_top_k, 8))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    db_path = out / "retention.sqlite"
    pack_path = out / "retention.eqbase"
    ir_path = out / "teacher-behaviour.eqir.json"
    probe_report_path = out / "teacher-probe-report.json"
    report_path = out / "retention-report.json"
    for p in [db_path, pack_path, ir_path, probe_report_path, report_path]:
        if p.exists():
            p.unlink()

    t0 = time.perf_counter()
    doc, probe_report, probes = extract_teacher_behaviour(
        backend, prompts, source_id, source_version, cfg
    )
    teacher_probe_elapsed_ms = (time.perf_counter() - t0) * 1000.0
    errors = [x for x in validate_document(doc) if x.severity == "error"]
    if errors:
        raise ValueError("teacher behaviour IR failed validation: " + "; ".join(x.message for x in errors))
    save_document(doc, ir_path)
    probe_report_path.write_text(json.dumps(probe_report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with CanonicalDB(db_path) as db:
        for node in doc.nodes:
            db.upsert_node(node)
        for evidence in doc.evidence:
            db.add_evidence(evidence)
        db.commit()
        # A pass-through compression run gives the runtime pack a concrete physical
        # view without introducing discovered macros into this first retention test.
        compression = compile_runtime_compression(
            db.db,
            MacroDiscoveryPolicy(
                min_pattern_size=2,
                max_pattern_size=4,
                min_occurrences=1_000_000,
                min_parent_structures=1_000_000,
                definition_overhead=1.0,
                min_net_savings=1.0,
                min_compression_ratio=0.0,
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
            persist=True,
            materialize=True,
        )
        retention_relation_policy = CompilePolicy(min_score=0.0, min_confidence=0.0, min_sources=1)
        manifest = compile_base_pack(
            db.db,
            pack_path,
            pack_id=pack_id,
            pack_version=pack_version,
            compression_run_id=compression.compression_run_id,
            include_metadata=include_pack_metadata,
            persist=True,
            verify_roundtrip=True,
            relation_policy=retention_relation_policy,
        )

    direct_cases, direct_eval_ms = benchmark_teacher_probes_against_pack(
        pack_path, probes, cfg, cognition_policy=_direct_replay_policy(policy, runtime_top_k),
        teacher_top_k=cfg.top_k, runtime_top_k=runtime_top_k,
    )
    cases, full_eval_ms = benchmark_teacher_probes_against_pack(
        pack_path, probes, cfg, cognition_policy=policy,
        teacher_top_k=cfg.top_k, runtime_top_k=runtime_top_k,
    )
    runtime_eval_elapsed_ms = direct_eval_ms + full_eval_ms
    direct_summary = _summarize_cases(direct_cases)
    summary = _summarize_cases(cases)
    n = len(cases)
    top1_matches = sum(1 for x in cases if x.top1_match)
    backend_manifest = dict(backend.manifest() or {})
    teacher_bytes = _directory_bytes(backend_manifest.get("model_dir"))
    pack_bytes = pack_path.stat().st_size

    report = BehaviourRetentionReport(
        format=RETENTION_FORMAT,
        version=RETENTION_VERSION,
        evaluation_scope="compiled_probe_replay",
        generalization_claim=False,
        runtime_scores_are_calibrated_probabilities=False,
        cases=n,
        teacher_top_k=cfg.top_k,
        runtime_top_k=runtime_top_k,
        top1_retention=summary["top1_retention"],
        mean_teacher_topk_recall=summary["mean_teacher_topk_recall"],
        mean_runtime_precision_against_teacher_topk=summary["mean_runtime_precision_against_teacher_topk"],
        mean_teacher_probability_mass_captured_by_probe=summary["mean_teacher_probability_mass_captured_by_probe"],
        mean_teacher_probability_mass_retained_in_runtime_topk=summary["mean_teacher_probability_mass_retained_in_runtime_topk"],
        mean_ndcg_at_k=summary["mean_ndcg_at_k"],
        mean_conditional_js_divergence=summary["mean_conditional_js_divergence"],
        mean_conditional_total_variation=summary["mean_conditional_total_variation"],
        direct_replay=direct_summary,
        unresolved_contexts=summary["unresolved_contexts"],
        no_runtime_predictions=summary["no_runtime_predictions"],
        average_recall_nodes=summary["average_recall_nodes"],
        average_relation_checks=summary["average_relation_checks"],
        teacher_probe_elapsed_ms=teacher_probe_elapsed_ms,
        runtime_eval_elapsed_ms=runtime_eval_elapsed_ms,
        pack_bytes=pack_bytes,
        pack_nodes=manifest.node_count,
        pack_relations=manifest.relation_count,
        compiled_prediction_evidence=probe_report.prediction_evidence,
        bytes_per_compiled_prompt=(pack_bytes / n if n else 0.0),
        bytes_per_retained_top1=(pack_bytes / top1_matches if top1_matches else None),
        teacher_model_bytes=teacher_bytes,
        pack_to_teacher_storage_ratio=(pack_bytes / teacher_bytes if teacher_bytes else None),
        backend_manifest=backend_manifest,
        probe_report=probe_report.to_dict(),
        pack_manifest=manifest.to_dict(),
        relation_compile_policy=asdict(retention_relation_policy),
        cases_detail=cases,
        notes=[
            "This is compiled-probe replay retention, not held-out language generalization.",
            "direct_replay disables associative recall depth to isolate compiler/serialization retention; top-level metrics use the configured full runtime policy.",
            "The retention pack uses relation min_score=0 and min_confidence=0 so accepted teacher top-k evidence is not silently pruned by production relation thresholds.",
            "Teacher probability mass is measured on accepted top-k candidates after probe filtering.",
            "Equinox runtime scores are ranking scores, not calibrated probabilities; JS/TV therefore compare distributions only after conditional top-k renormalization.",
            "Host wall-clock timings are diagnostics only and are not mobile energy or latency claims.",
        ],
    )
    report_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
