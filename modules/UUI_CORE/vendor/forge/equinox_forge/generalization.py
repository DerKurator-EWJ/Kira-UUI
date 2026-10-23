from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json
import shutil
import time

from .behaviour_retention import (
    BehaviourRetentionCase,
    _direct_replay_policy,
    _summarize_cases,
    benchmark_teacher_probes_against_pack,
)
from .canonical import normalize_surface
from .compiler import CompilePolicy
from .database import CanonicalDB
from .macro_discovery import MacroDiscoveryPolicy
from .runtime_compression import MacroPromotionPolicy, compile_runtime_compression
from .runtime_pack import compile_base_pack, load_base_pack
from .runtime_cognition import RuntimeCognition, RuntimeCognitionPolicy
from .runtime_query import RuntimeIndex
from .teacher_behavior import (
    BehaviourProbeConfig,
    CausalTeacherBackend,
    TeacherProbe,
    extract_teacher_behaviour,
)
from .validate import validate_document
from .io import save_document

GENERALIZATION_FORMAT = "equinox-forge-heldout-generalization-v1"
GENERALIZATION_VERSION = "0.19"


@dataclass
class ContextCoverageCase:
    prompt: str
    bucket: str
    resolved_context_surfaces: list[str]
    resolved_context_width: int
    prompt_token_count: int
    resolved_suffix_ratio: float
    known_lexical_tokens: int
    lexical_tokens: int
    known_token_coverage: float
    novel_tokens: int
    token_novelty_rate: float
    teacher_top1: str | None
    teacher_top1_known_in_pack: bool
    top1_match: bool
    runtime_top1: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HeldOutGeneralizationReport:
    format: str
    version: str
    evaluation_scope: str
    generalization_claim: bool
    heldout_transfer_claim: bool
    train_prompts: int
    heldout_prompts: int
    exact_split_overlap: int
    train_dataset_sha256: str
    heldout_dataset_sha256: str
    pack_bytes: int
    pack_nodes: int
    pack_relations: int
    compiled_prediction_evidence: int
    train_replay: dict[str, Any]
    heldout_direct_replay: dict[str, Any]
    heldout_full_runtime: dict[str, Any]
    top1_generalization_gap: float
    teacher_mass_generalization_gap: float
    heldout_resolved_context_rate: float
    heldout_proper_suffix_transfer_rate: float
    heldout_lexical_only_rate: float
    heldout_unresolved_rate: float
    heldout_teacher_top1_known_in_pack_rate: float
    mean_resolved_suffix_ratio: float
    mean_known_token_coverage: float
    mean_token_novelty_rate: float
    heldout_exact_context_nodes_in_pack: int
    coverage_buckets: dict[str, dict[str, Any]]
    teacher_probe_elapsed_ms: float
    runtime_eval_elapsed_ms: float
    backend_manifest: dict[str, Any]
    train_probe_report: dict[str, Any]
    heldout_probe_report: dict[str, Any]
    pack_manifest: dict[str, Any]
    relation_compile_policy: dict[str, Any]
    coverage_detail: list[ContextCoverageCase]
    heldout_cases_detail: list[BehaviourRetentionCase]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["coverage_detail"] = [x.to_dict() for x in self.coverage_detail]
        d["heldout_cases_detail"] = [x.to_dict() for x in self.heldout_cases_detail]
        return d


def _normalized_prompts(prompts: Iterable[str]) -> tuple[list[str], int]:
    out: list[str] = []
    seen: set[str] = set()
    duplicates = 0
    for raw in prompts:
        text = normalize_surface(str(raw)).strip("\r\n ")
        if not text:
            continue
        if text in seen:
            duplicates += 1
            continue
        seen.add(text)
        out.append(text)
    return out, duplicates


def _dataset_hash(prompts: list[str]) -> str:
    payload = "\n".join(prompts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tokenize_runtime(text: str) -> list[str]:
    return [x for x in text.strip().split() if x]


def _pass_through_compression(db) -> Any:
    return compile_runtime_compression(
        db,
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


def _coverage_cases(
    pack_path: str | Path,
    train_prompts: list[str],
    probes: list[TeacherProbe],
    retention_cases: list[BehaviourRetentionCase],
    policy: RuntimeCognitionPolicy,
) -> tuple[list[ContextCoverageCase], int]:
    pack = load_base_pack(pack_path)
    cog = RuntimeCognition(pack, policy)
    index = RuntimeIndex(pack)
    train_set = set(train_prompts)
    train_tokens = {tok for p in train_prompts for tok in _tokenize_runtime(p)}
    by_prompt = {c.prompt: c for c in retention_cases}

    exact_context_nodes = 0
    rows: list[ContextCoverageCase] = []
    for probe in probes:
        prompt = normalize_surface(probe.prompt).strip()
        pred = cog.predict(prompt)
        tokens = _tokenize_runtime(prompt)
        resolved_surfaces: list[str] = []
        resolved_width = 0
        exact_node_present = False
        for nid in pred.resolved_context_ids:
            node = index.node_by_id.get(nid)
            if node is None or not node.surface:
                continue
            surf = normalize_surface(node.surface).strip()
            resolved_surfaces.append(surf)
            width = len(_tokenize_runtime(surf))
            resolved_width = max(resolved_width, width)
            if surf == prompt:
                exact_node_present = True
        if exact_node_present:
            exact_context_nodes += 1

        if prompt in train_set or exact_node_present:
            bucket = "exact_or_leaked"
        elif resolved_surfaces:
            proper_suffix = False
            for surf in resolved_surfaces:
                st = _tokenize_runtime(surf)
                if st and len(st) < len(tokens) and tokens[-len(st):] == st:
                    proper_suffix = True
                    break
            bucket = "proper_suffix_transfer" if proper_suffix else "resolved_structural"
        elif pred.lexical_seed_ids:
            bucket = "lexical_only"
        else:
            bucket = "unresolved"

        known_lexical = sum(1 for t in tokens if t in index.surface_to_nodes)
        novel = sum(1 for t in tokens if t not in train_tokens)
        retained = by_prompt.get(probe.prompt)
        teacher_top1 = retained.teacher_top1 if retained else None
        teacher_top1_known = bool(teacher_top1 and teacher_top1 in index.surface_to_nodes)
        rows.append(ContextCoverageCase(
            prompt=probe.prompt,
            bucket=bucket,
            resolved_context_surfaces=sorted(set(resolved_surfaces)),
            resolved_context_width=resolved_width,
            prompt_token_count=len(tokens),
            resolved_suffix_ratio=(resolved_width / len(tokens) if tokens else 0.0),
            known_lexical_tokens=known_lexical,
            lexical_tokens=len(tokens),
            known_token_coverage=(known_lexical / len(tokens) if tokens else 0.0),
            novel_tokens=novel,
            token_novelty_rate=(novel / len(tokens) if tokens else 0.0),
            teacher_top1=teacher_top1,
            teacher_top1_known_in_pack=teacher_top1_known,
            top1_match=(retained.top1_match if retained else False),
            runtime_top1=(retained.runtime_top1 if retained else None),
        ))
    return rows, exact_context_nodes


def _bucket_summaries(rows: list[ContextCoverageCase]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for bucket in sorted({x.bucket for x in rows}):
        subset = [x for x in rows if x.bucket == bucket]
        n = len(subset)
        out[bucket] = {
            "cases": n,
            "share": n / len(rows) if rows else 0.0,
            "top1_retention": sum(1 for x in subset if x.top1_match) / n if n else 0.0,
            "teacher_top1_known_in_pack_rate": sum(1 for x in subset if x.teacher_top1_known_in_pack) / n if n else 0.0,
            "mean_resolved_suffix_ratio": sum(x.resolved_suffix_ratio for x in subset) / n if n else 0.0,
            "mean_known_token_coverage": sum(x.known_token_coverage for x in subset) / n if n else 0.0,
        }
    return out


def compile_and_benchmark_heldout_generalization(
    backend: CausalTeacherBackend,
    train_prompts: Iterable[str],
    heldout_prompts: Iterable[str],
    out_dir: str | Path,
    source_id: str,
    source_version: str,
    *,
    probe_cfg: BehaviourProbeConfig | None = None,
    cognition_policy: RuntimeCognitionPolicy | None = None,
    runtime_top_k: int | None = None,
    pack_id: str = "equinox-heldout-generalization",
    pack_version: int = 19,
    include_pack_metadata: bool = False,
    strict_split: bool = True,
) -> HeldOutGeneralizationReport:
    """Compile only training prompts and evaluate disjoint held-out prompts.

    This benchmark demonstrates held-out *context transfer* through already compiled
    Equinox structure. It does not claim broad language-model generalization: the
    runtime cannot invent knowledge that is absent from the compiled BASE.
    """
    cfg = probe_cfg or BehaviourProbeConfig()
    cfg.validate()
    train, train_dupes = _normalized_prompts(train_prompts)
    heldout, heldout_dupes = _normalized_prompts(heldout_prompts)
    if not train or not heldout:
        raise ValueError("both train and heldout splits require at least one non-empty prompt")
    overlap = sorted(set(train) & set(heldout))
    if strict_split and overlap:
        raise ValueError("train/heldout exact prompt leakage: " + "; ".join(overlap[:8]))

    runtime_top_k = int(runtime_top_k or cfg.top_k)
    policy = cognition_policy or RuntimeCognitionPolicy(max_predictions=max(runtime_top_k, 8))
    policy.validate()

    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    eval_dir = out / "evaluation_only"
    eval_dir.mkdir(parents=True, exist_ok=True)
    db_path = out / "train.sqlite"
    pack_path = out / "train.eqbase"

    tprobe = time.perf_counter()
    train_doc, train_probe_report, train_probes = extract_teacher_behaviour(
        backend, train, source_id, source_version, cfg
    )
    # Held-out teacher data is obtained for scoring only and is NEVER inserted into
    # the compile DB. Keeping it in evaluation_only makes the boundary auditable.
    heldout_doc, heldout_probe_report, heldout_probes = extract_teacher_behaviour(
        backend, heldout, source_id + ":heldout-reference", source_version, cfg
    )
    teacher_probe_elapsed_ms = (time.perf_counter() - tprobe) * 1000.0
    for label, doc in [("train", train_doc), ("heldout", heldout_doc)]:
        errors = [x for x in validate_document(doc) if x.severity == "error"]
        if errors:
            raise ValueError(f"{label} teacher IR failed validation: " + "; ".join(x.message for x in errors))
    save_document(train_doc, out / "train-teacher-behaviour.eqir.json")
    save_document(heldout_doc, eval_dir / "heldout-teacher-reference.eqir.json")
    (out / "train-teacher-probe-report.json").write_text(
        json.dumps(train_probe_report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (eval_dir / "heldout-teacher-probe-report.json").write_text(
        json.dumps(heldout_probe_report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    split_manifest = {
        "train_prompts": train,
        "heldout_prompts": heldout,
        "train_sha256": _dataset_hash(train),
        "heldout_sha256": _dataset_hash(heldout),
        "exact_overlap": overlap,
        "strict_split": strict_split,
        "train_duplicates_removed": train_dupes,
        "heldout_duplicates_removed": heldout_dupes,
        "heldout_reference_compiled": False,
    }
    (out / "split-manifest.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    relation_policy = CompilePolicy(min_score=0.0, min_confidence=0.0, min_sources=1)
    with CanonicalDB(db_path) as db:
        for node in train_doc.nodes:
            db.upsert_node(node)
        for evidence in train_doc.evidence:
            db.add_evidence(evidence)
        db.commit()
        compression = _pass_through_compression(db.db)
        manifest = compile_base_pack(
            db.db,
            pack_path,
            pack_id=pack_id,
            pack_version=pack_version,
            compression_run_id=compression.compression_run_id,
            include_metadata=include_pack_metadata,
            persist=True,
            verify_roundtrip=True,
            relation_policy=relation_policy,
        )

    teval = time.perf_counter()
    train_cases, _ = benchmark_teacher_probes_against_pack(
        pack_path, train_probes, cfg,
        cognition_policy=policy,
        teacher_top_k=cfg.top_k,
        runtime_top_k=runtime_top_k,
    )
    heldout_direct_cases, _ = benchmark_teacher_probes_against_pack(
        pack_path, heldout_probes, cfg,
        cognition_policy=_direct_replay_policy(policy, runtime_top_k),
        teacher_top_k=cfg.top_k,
        runtime_top_k=runtime_top_k,
    )
    heldout_cases, _ = benchmark_teacher_probes_against_pack(
        pack_path, heldout_probes, cfg,
        cognition_policy=policy,
        teacher_top_k=cfg.top_k,
        runtime_top_k=runtime_top_k,
    )
    runtime_eval_elapsed_ms = (time.perf_counter() - teval) * 1000.0

    train_summary = _summarize_cases(train_cases)
    heldout_direct = _summarize_cases(heldout_direct_cases)
    heldout_summary = _summarize_cases(heldout_cases)
    coverage, exact_nodes = _coverage_cases(pack_path, train, heldout_probes, heldout_cases, policy)
    n = len(coverage)
    buckets = _bucket_summaries(coverage)
    count = lambda b: sum(1 for x in coverage if x.bucket == b)

    backend_manifest = dict(backend.manifest() or {})
    report = HeldOutGeneralizationReport(
        format=GENERALIZATION_FORMAT,
        version=GENERALIZATION_VERSION,
        evaluation_scope="strict_heldout_context_transfer",
        generalization_claim=False,
        heldout_transfer_claim=True,
        train_prompts=len(train),
        heldout_prompts=len(heldout),
        exact_split_overlap=len(overlap),
        train_dataset_sha256=_dataset_hash(train),
        heldout_dataset_sha256=_dataset_hash(heldout),
        pack_bytes=manifest.file_bytes,
        pack_nodes=manifest.node_count,
        pack_relations=manifest.relation_count,
        compiled_prediction_evidence=train_probe_report.prediction_evidence,
        train_replay=train_summary,
        heldout_direct_replay=heldout_direct,
        heldout_full_runtime=heldout_summary,
        top1_generalization_gap=train_summary["top1_retention"] - heldout_summary["top1_retention"],
        teacher_mass_generalization_gap=(
            train_summary["mean_teacher_probability_mass_retained_in_runtime_topk"]
            - heldout_summary["mean_teacher_probability_mass_retained_in_runtime_topk"]
        ),
        heldout_resolved_context_rate=(sum(1 for x in coverage if x.bucket in {"proper_suffix_transfer", "resolved_structural", "exact_or_leaked"}) / n if n else 0.0),
        heldout_proper_suffix_transfer_rate=(count("proper_suffix_transfer") / n if n else 0.0),
        heldout_lexical_only_rate=(count("lexical_only") / n if n else 0.0),
        heldout_unresolved_rate=(count("unresolved") / n if n else 0.0),
        heldout_teacher_top1_known_in_pack_rate=(sum(1 for x in coverage if x.teacher_top1_known_in_pack) / n if n else 0.0),
        mean_resolved_suffix_ratio=(sum(x.resolved_suffix_ratio for x in coverage) / n if n else 0.0),
        mean_known_token_coverage=(sum(x.known_token_coverage for x in coverage) / n if n else 0.0),
        mean_token_novelty_rate=(sum(x.token_novelty_rate for x in coverage) / n if n else 0.0),
        heldout_exact_context_nodes_in_pack=exact_nodes,
        coverage_buckets=buckets,
        teacher_probe_elapsed_ms=teacher_probe_elapsed_ms,
        runtime_eval_elapsed_ms=runtime_eval_elapsed_ms,
        backend_manifest=backend_manifest,
        train_probe_report=train_probe_report.to_dict(),
        heldout_probe_report=heldout_probe_report.to_dict(),
        pack_manifest=manifest.to_dict(),
        relation_compile_policy=asdict(relation_policy),
        coverage_detail=coverage,
        heldout_cases_detail=heldout_cases,
        notes=[
            "Only train-split teacher evidence is compiled into train.sqlite and train.eqbase; held-out teacher probes are evaluation-only references.",
            "Exact normalized train/held-out prompt overlap is rejected by default.",
            "A proper_suffix_transfer case is held out as a full prompt but reuses a shorter predictive context that was compiled from the train split.",
            "This benchmark supports claims about held-out structural/context transfer only; it is not evidence of broad language-model generalization or novel factual reasoning.",
            "The teacher top-1-known-in-pack rate separates missing target vocabulary/knowledge from context-resolution failure.",
            "Direct held-out replay disables associative recall depth but still permits deterministic longest-known-suffix context resolution; full runtime additionally measures bounded recall effects.",
            "Equinox runtime scores are ranking scores, not calibrated teacher probabilities.",
            "Host wall-clock timings are diagnostics only and are not mobile energy or latency claims.",
        ],
    )
    (out / "generalization-report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
