from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json
import shutil
import time

from .behaviour_retention import benchmark_teacher_probes_against_pack, _summarize_cases
from .canonical import normalize_surface
from .compiler import CompilePolicy
from .database import CanonicalDB
from .frame_induction import FrameInductionPolicy, induce_relation_frames, persist_frame_document
from .generalization import _pass_through_compression
from .io import save_document
from .runtime_cognition import RuntimeCognition, RuntimeCognitionPolicy
from .runtime_pack import compile_base_pack, load_base_pack
from .teacher_behavior import BehaviourProbeConfig, CausalTeacherBackend, extract_teacher_behaviour
from .validate import validate_document
from .ir import ForgeDocument

FRAME_BENCHMARK_FORMAT = "equinox-forge-frame-schema-benchmark-v1"
FRAME_BENCHMARK_VERSION = "0.24"


@dataclass(frozen=True)
class FrameEvalCase:
    prompt: str
    expected_frame_applicable: bool | None = None
    note: str = ""


@dataclass
class FrameEvalDetail:
    prompt: str
    teacher_top1: str | None
    frame_only_top1: str | None
    no_frame_top1: str | None
    open_frame_top1: str | None
    frame_matched: bool
    frame_admission: list[dict[str, Any]]
    frame_surfaces: list[str]
    frame_only_top1_match: bool
    expected_frame_applicable: bool | None
    applicability_correct: bool | None
    note: str

    def to_dict(self) -> dict[str, Any]: return asdict(self)


@dataclass
class FrameSchemaBenchmarkReport:
    format: str
    version: str
    train_prompts: int
    eval_prompts: int
    exact_prompt_overlap: int
    train_sha256: str
    eval_sha256: str
    induced_frames: int
    induced_prediction_evidence: int
    pack_bytes: int
    pack_nodes: int
    pack_relations: int
    frame_only: dict[str, Any]
    no_frame: dict[str, Any]
    open_frame_baseline: dict[str, Any]
    full_runtime: dict[str, Any]
    frame_match_rate: float
    frame_applicability_precision: float | None
    frame_applicability_recall: float | None
    semantic_admission_enabled: bool
    support_documents: int
    details: list[FrameEvalDetail]
    frame_induction: dict[str, Any]
    backend_manifest: dict[str, Any]
    pack_manifest: dict[str, Any]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["details"] = [x.to_dict() for x in self.details]
        return d


def _norm(text: str) -> str:
    return normalize_surface(str(text)).strip("\r\n ")


def _hash_rows(rows: list[Any]) -> str:
    raw = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _frame_only_policy(base: RuntimeCognitionPolicy, enabled: bool) -> RuntimeCognitionPolicy:
    return replace(
        base,
        enable_exact_context=False,
        enable_suffix_context=False,
        enable_lexical_seeds=False,
        enable_parent_routing=False,
        enable_relation_recall=False,
        enable_transition_prediction=False,
        enable_recalled_context_prediction=False,
        enable_direct_context_prediction=True,
        enable_frame_schema_matching=enabled,
    )


def compile_and_benchmark_frame_schemas(
    backend: CausalTeacherBackend,
    train_prompts: Iterable[str],
    eval_cases: Iterable[FrameEvalCase | str],
    out_dir: str | Path,
    source_id: str,
    source_version: str,
    *,
    probe_cfg: BehaviourProbeConfig | None = None,
    frame_policy: FrameInductionPolicy | None = None,
    cognition_policy: RuntimeCognitionPolicy | None = None,
    runtime_top_k: int | None = None,
    support_documents: Iterable[ForgeDocument] | None = None,
    pack_id: str = "equinox-frame-schema-benchmark",
    pack_version: int = 24,
) -> FrameSchemaBenchmarkReport:
    cfg = probe_cfg or BehaviourProbeConfig()
    cfg.validate()
    fp = frame_policy or FrameInductionPolicy()
    fp.validate()
    policy = cognition_policy or RuntimeCognitionPolicy(max_predictions=max(int(runtime_top_k or cfg.top_k), 8))
    policy.validate()
    runtime_top_k = int(runtime_top_k or cfg.top_k)
    support_docs = list(support_documents or [])
    for i, doc in enumerate(support_docs):
        errors = [x for x in validate_document(doc) if x.severity == "error"]
        if errors:
            raise ValueError(f"support document {i} failed validation: " + "; ".join(x.message for x in errors))

    train = []
    seen = set()
    for p in train_prompts:
        x = _norm(p)
        if x and x not in seen:
            seen.add(x); train.append(x)
    eval_rows: list[FrameEvalCase] = []
    for row in eval_cases:
        if isinstance(row, FrameEvalCase):
            prompt = _norm(row.prompt)
            if prompt: eval_rows.append(FrameEvalCase(prompt, row.expected_frame_applicable, row.note))
        else:
            prompt = _norm(str(row))
            if prompt: eval_rows.append(FrameEvalCase(prompt))
    if not train or not eval_rows:
        raise ValueError("train_prompts and eval_cases must both be non-empty")
    overlap = sorted(set(train) & {x.prompt for x in eval_rows})
    if overlap:
        raise ValueError("frame benchmark requires held-out eval prompts; overlap: " + ", ".join(overlap[:8]))

    out = Path(out_dir)
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    eval_dir = out / "evaluation_only"
    eval_dir.mkdir(parents=True, exist_ok=True)
    db_path = out / "train.sqlite"
    pack_path = out / "train-with-frames.eqbase"

    t0 = time.perf_counter()
    train_doc, train_probe_report, _ = extract_teacher_behaviour(backend, train, source_id, source_version, cfg)
    eval_doc, eval_probe_report, eval_probes = extract_teacher_behaviour(
        backend, [x.prompt for x in eval_rows], source_id + ":frame-eval-reference", source_version, cfg
    )
    teacher_ms = (time.perf_counter() - t0) * 1000.0
    for label, doc in (("train", train_doc), ("eval", eval_doc)):
        errors = [x for x in validate_document(doc) if x.severity == "error"]
        if errors: raise ValueError(f"{label} teacher IR failed validation: " + "; ".join(x.message for x in errors))
    save_document(train_doc, out / "train-teacher-behaviour.eqir.json")
    save_document(eval_doc, eval_dir / "frame-eval-teacher-reference.eqir.json")
    support_dir = out / "support_knowledge"
    if support_docs:
        support_dir.mkdir(parents=True, exist_ok=True)
        for i, doc in enumerate(support_docs):
            save_document(doc, support_dir / f"support-{i:02d}.eqir.json")

    relation_policy = CompilePolicy(min_score=0.0, min_confidence=0.0, min_sources=1)
    with CanonicalDB(db_path) as db:
        for n in train_doc.nodes: db.upsert_node(n)
        for e in train_doc.evidence: db.add_evidence(e)
        for doc in support_docs:
            for n in doc.nodes: db.upsert_node(n)
            for e in doc.evidence: db.add_evidence(e)
        db.commit()
        frame_doc, frame_report = induce_relation_frames(db.db, fp)
        frame_errors = [x for x in validate_document(frame_doc) if x.severity == "error"]
        if frame_errors:
            raise ValueError("induced frame IR failed validation: " + "; ".join(x.message for x in frame_errors))
        persist_frame_document(db.db, frame_doc)
        db.db.commit()
        save_document(frame_doc, out / "induced-frames.eqir.json")
        (out / "frame-induction-report.json").write_text(
            json.dumps(frame_report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        compression = _pass_through_compression(db.db)
        manifest = compile_base_pack(
            db.db, pack_path, pack_id=pack_id, pack_version=pack_version,
            compression_run_id=compression.compression_run_id,
            include_metadata=True, persist=True, verify_roundtrip=True,
            relation_policy=relation_policy,
        )

    t1 = time.perf_counter()
    frame_cases, _ = benchmark_teacher_probes_against_pack(
        pack_path, eval_probes, cfg, cognition_policy=_frame_only_policy(policy, True),
        teacher_top_k=cfg.top_k, runtime_top_k=runtime_top_k,
    )
    no_frame_cases, _ = benchmark_teacher_probes_against_pack(
        pack_path, eval_probes, cfg, cognition_policy=_frame_only_policy(policy, False),
        teacher_top_k=cfg.top_k, runtime_top_k=runtime_top_k,
    )
    open_frame_policy = replace(policy, enable_semantic_slot_admission=False)
    open_frame_cases, _ = benchmark_teacher_probes_against_pack(
        pack_path, eval_probes, cfg, cognition_policy=_frame_only_policy(open_frame_policy, True),
        teacher_top_k=cfg.top_k, runtime_top_k=runtime_top_k,
    )
    full_cases, _ = benchmark_teacher_probes_against_pack(
        pack_path, eval_probes, cfg, cognition_policy=policy,
        teacher_top_k=cfg.top_k, runtime_top_k=runtime_top_k,
    )
    runtime_ms = (time.perf_counter() - t1) * 1000.0

    pack = load_base_pack(pack_path)
    idx = pack.node_by_id()
    frame_by_prompt = {x.prompt: x for x in frame_cases}
    no_frame_by_prompt = {x.prompt: x for x in no_frame_cases}
    open_frame_by_prompt = {x.prompt: x for x in open_frame_cases}
    eval_probe_by_prompt = {x.prompt: x for x in eval_probes}
    frame_cog = RuntimeCognition(pack, _frame_only_policy(policy, True))
    detail: list[FrameEvalDetail] = []
    tp = fp_count = fn = 0
    for case in eval_rows:
        pred = frame_cog.predict(case.prompt)
        frame_nodes = [idx[nid] for nid in pred.resolved_context_ids if nid in idx and (idx[nid].metadata or {}).get("role") == "induced_frame_schema"]
        matched = bool(frame_nodes)
        if case.expected_frame_applicable is True:
            if matched: tp += 1
            else: fn += 1
        elif case.expected_frame_applicable is False and matched:
            fp_count += 1
        fr = frame_by_prompt[case.prompt]
        nr = no_frame_by_prompt[case.prompt]
        orow = open_frame_by_prompt[case.prompt]
        teacher_top1 = fr.teacher_top1
        frame_top1 = fr.runtime_top1
        no_frame_top1 = nr.runtime_top1
        open_frame_top1 = orow.runtime_top1
        applicability_correct = None if case.expected_frame_applicable is None else (matched == case.expected_frame_applicable)
        detail.append(FrameEvalDetail(
            prompt=case.prompt,
            teacher_top1=teacher_top1,
            frame_only_top1=frame_top1,
            no_frame_top1=no_frame_top1,
            open_frame_top1=open_frame_top1,
            frame_matched=matched,
            frame_admission=list(pred.frame_admission),
            frame_surfaces=[n.surface or n.node_id for n in frame_nodes],
            frame_only_top1_match=(frame_top1 == teacher_top1 and teacher_top1 is not None),
            expected_frame_applicable=case.expected_frame_applicable,
            applicability_correct=applicability_correct,
            note=case.note,
        ))

    precision = (tp / (tp + fp_count)) if (tp + fp_count) else None
    recall = (tp / (tp + fn)) if (tp + fn) else None
    report = FrameSchemaBenchmarkReport(
        format=FRAME_BENCHMARK_FORMAT,
        version=FRAME_BENCHMARK_VERSION,
        train_prompts=len(train), eval_prompts=len(eval_rows), exact_prompt_overlap=0,
        train_sha256=_hash_rows(train),
        eval_sha256=_hash_rows([asdict(x) for x in eval_rows]),
        induced_frames=frame_report.accepted_frames,
        induced_prediction_evidence=frame_report.generated_prediction_evidence,
        pack_bytes=manifest.file_bytes, pack_nodes=manifest.node_count, pack_relations=manifest.relation_count,
        frame_only=_summarize_cases(frame_cases),
        no_frame=_summarize_cases(no_frame_cases),
        open_frame_baseline=_summarize_cases(open_frame_cases),
        full_runtime=_summarize_cases(full_cases),
        frame_match_rate=(sum(1 for x in detail if x.frame_matched) / len(detail) if detail else 0.0),
        frame_applicability_precision=precision,
        frame_applicability_recall=recall,
        semantic_admission_enabled=bool(policy.enable_semantic_slot_admission),
        support_documents=len(support_docs),
        details=detail,
        frame_induction=frame_report.to_dict(),
        backend_manifest={**dict(backend.manifest() or {}), "teacher_probe_elapsed_ms": teacher_ms, "runtime_eval_elapsed_ms": runtime_ms,
                          "train_probe_report": train_probe_report.to_dict(), "eval_probe_report": eval_probe_report.to_dict()},
        pack_manifest=manifest.to_dict(),
        notes=[
            "Evaluation teacher probes are reference-only and are never inserted into train.sqlite or train-with-frames.eqbase.",
            "frame_only disables exact/suffix lookup, lexical seeds, parent routing, relation recall, transitions and recalled-context prediction; any surviving prediction therefore comes from an induced frame schema.",
            "Open text slots are structural wildcards, not semantic type guarantees. A frame may overgeneralize to a novel filler outside the intended latent class.",
            "Applicability labels, when supplied, are evaluation-only annotations and are never compiled into the BASE.",
            "v0.24 can enforce compiled semantic slot admission and optionally use a latent prototype membership edge instead of pairwise filler checks. The open_frame_baseline disables the semantic gate on the same BASE.",
            "Support documents may contain independent lexical/semantic knowledge, but evaluation prompts and teacher answers remain reference-only.",
        ],
    )
    (out / "frame-schema-benchmark-report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def load_frame_eval_jsonl(path: str | Path) -> list[FrameEvalCase]:
    out: list[FrameEvalCase] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not row.get("prompt"):
            raise ValueError(f"{path}:{lineno}: expected object with prompt")
        applicable = row.get("expected_frame_applicable")
        if applicable is not None and not isinstance(applicable, bool):
            raise ValueError(f"{path}:{lineno}: expected_frame_applicable must be boolean or null")
        out.append(FrameEvalCase(str(row["prompt"]), applicable, str(row.get("note", ""))))
    return out
