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
from .generalization import _pass_through_compression, _tokenize_runtime
from .io import save_document
from .ir import ForgeDocument
from .runtime_cognition import RuntimeCognition, RuntimeCognitionPolicy
from .runtime_pack import compile_base_pack, load_base_pack
from .runtime_query import RuntimeIndex
from .teacher_behavior import BehaviourProbeConfig, CausalTeacherBackend, extract_teacher_behaviour
from .validate import validate_document

LADDER_FORMAT = "equinox-forge-generalization-ladder-v1"
LADDER_VERSION = "0.20"

LEVELS = {
    0: "exact_replay_control",
    1: "known_suffix_in_new_outer_context",
    2: "novel_recombination_known_parts",
    3: "novel_relation_order_known_lexicon",
    4: "novel_entity_known_relational_frame",
    5: "heldout_relation_family",
}


@dataclass(frozen=True)
class LadderTrainCase:
    prompt: str
    family: str = ""


@dataclass(frozen=True)
class LadderEvalCase:
    prompt: str
    level: int
    family: str = ""
    note: str = ""


@dataclass
class LadderStructuralDiagnostics:
    prompt: str
    requested_level: int
    level_name: str
    exact_in_train: bool
    proper_train_suffix: str | None
    prompt_tokens: list[str]
    known_tokens: int
    novel_tokens: int
    token_coverage: float
    bigrams: int
    known_bigrams: int
    bigram_coverage: float
    family_seen_in_train: bool
    resolved_context_surfaces: list[str]
    resolved_context_width: int
    runtime_resolution_kind: str
    structural_contract_ok: bool
    contract_violations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LadderLevelReport:
    level: int
    level_name: str
    cases: int
    top1_retention: float
    mean_teacher_topk_recall: float
    mean_teacher_probability_mass_retained: float
    mean_ndcg_at_k: float
    mean_conditional_js_divergence: float
    resolved_context_rate: float
    mean_token_coverage: float
    mean_bigram_coverage: float
    structural_contract_pass_rate: float
    direct_replay: dict[str, Any]
    full_runtime: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GeneralizationLadderReport:
    format: str
    version: str
    evaluation_scope: str
    generalization_claim: bool
    train_cases: int
    evaluation_cases: int
    heldout_cases: int
    exact_replay_controls: int
    train_dataset_sha256: str
    evaluation_dataset_sha256: str
    pack_bytes: int
    pack_nodes: int
    pack_relations: int
    compiled_prediction_evidence: int
    level_reports: list[LadderLevelReport]
    structural_diagnostics: list[LadderStructuralDiagnostics]
    cases_detail: list[dict[str, Any]]
    highest_level_with_any_top1_success: int | None
    highest_level_with_full_top1_success: int | None
    highest_contiguous_level_with_full_top1_success: int | None
    teacher_probe_elapsed_ms: float
    runtime_eval_elapsed_ms: float
    backend_manifest: dict[str, Any]
    pack_manifest: dict[str, Any]
    relation_compile_policy: dict[str, Any]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["level_reports"] = [x.to_dict() for x in self.level_reports]
        d["structural_diagnostics"] = [x.to_dict() for x in self.structural_diagnostics]
        return d


def _norm(text: str) -> str:
    return normalize_surface(str(text)).strip("\r\n ")


def _hash_rows(rows: Iterable[dict[str, Any]]) -> str:
    payload = json.dumps(list(rows), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}


def _longest_whole_train_suffix(prompt: str, train_prompts: set[str]) -> str | None:
    tokens = _tokenize_runtime(prompt)
    for width in range(len(tokens) - 1, 1, -1):
        suffix = " ".join(tokens[-width:])
        if suffix in train_prompts:
            return suffix
    return None


def _validate_eval_contract(
    case: LadderEvalCase,
    train_prompts: set[str],
    train_tokens: set[str],
    train_bigrams: set[tuple[str, str]],
    train_families: set[str],
) -> tuple[bool, list[str], dict[str, Any]]:
    if case.level not in LEVELS:
        return False, [f"unsupported ladder level: {case.level}"], {}
    prompt = _norm(case.prompt)
    tokens = _tokenize_runtime(prompt)
    exact = prompt in train_prompts
    suffix = _longest_whole_train_suffix(prompt, train_prompts)
    known = sum(1 for t in tokens if t in train_tokens)
    novel = len(tokens) - known
    pairs = _bigrams(tokens)
    known_pairs = pairs & train_bigrams
    fam_seen = bool(case.family and case.family in train_families)
    violations: list[str] = []

    if case.level == 0:
        if not exact: violations.append("L0 requires the exact prompt to be present in the compile/train split")
    else:
        if exact: violations.append("L1-L5 must not exactly overlap the compile/train split")

    if case.level == 1:
        if suffix is None: violations.append("L1 requires a proper suffix that is itself a compiled train prompt")
    elif case.level == 2:
        if novel: violations.append("L2 requires all lexical tokens to be known from train")
        if suffix is not None: violations.append("L2 must not reduce to L1 whole-prompt suffix reuse")
        if not known_pairs: violations.append("L2 requires at least one previously observed bigram/part relation")
    elif case.level == 3:
        if novel: violations.append("L3 requires a fully known lexicon")
        if suffix is not None: violations.append("L3 must not reduce to L1 whole-prompt suffix reuse")
        if known_pairs: violations.append("L3 requires novel local adjacency/order (no train bigram reuse)")
    elif case.level == 4:
        if novel < 1: violations.append("L4 requires at least one novel lexical entity/token")
        if not known_pairs: violations.append("L4 requires a known relational frame/bigram around the novel entity")
    elif case.level == 5:
        if not case.family: violations.append("L5 requires an explicit family label for auditable family holdout")
        if fam_seen: violations.append("L5 family must be absent from all train cases")

    return (not violations), violations, {
        "exact": exact,
        "suffix": suffix,
        "tokens": tokens,
        "known": known,
        "novel": novel,
        "pairs": pairs,
        "known_pairs": known_pairs,
        "family_seen": fam_seen,
    }


def _resolution_diagnostics(pack_path: str | Path, prompt: str, policy: RuntimeCognitionPolicy) -> tuple[list[str], int, str]:
    pack = load_base_pack(pack_path)
    cog = RuntimeCognition(pack, policy)
    index = RuntimeIndex(pack)
    pred = cog.predict(prompt)
    surfaces: list[str] = []
    width = 0
    norm_prompt = _norm(prompt)
    for nid in pred.resolved_context_ids:
        node = index.node_by_id.get(nid)
        if node is None or not node.surface:
            continue
        surf = _norm(node.surface)
        surfaces.append(surf)
        width = max(width, len(_tokenize_runtime(surf)))
    if any(x == norm_prompt for x in surfaces):
        kind = "exact"
    elif surfaces:
        kind = "suffix_or_structural"
    elif pred.lexical_seed_ids:
        kind = "lexical_only"
    else:
        kind = "unresolved"
    return sorted(set(surfaces)), width, kind


def _level_report(level: int, full_cases: list[BehaviourRetentionCase], direct_cases: list[BehaviourRetentionCase], diagnostics: list[LadderStructuralDiagnostics]) -> LadderLevelReport:
    full = _summarize_cases(full_cases)
    direct = _summarize_cases(direct_cases)
    n = len(full_cases)
    return LadderLevelReport(
        level=level,
        level_name=LEVELS[level],
        cases=n,
        top1_retention=full["top1_retention"],
        mean_teacher_topk_recall=full["mean_teacher_topk_recall"],
        mean_teacher_probability_mass_retained=full["mean_teacher_probability_mass_retained_in_runtime_topk"],
        mean_ndcg_at_k=full["mean_ndcg_at_k"],
        mean_conditional_js_divergence=full["mean_conditional_js_divergence"],
        resolved_context_rate=(sum(1 for x in full_cases if x.resolved_context) / n if n else 0.0),
        mean_token_coverage=(sum(x.token_coverage for x in diagnostics) / len(diagnostics) if diagnostics else 0.0),
        mean_bigram_coverage=(sum(x.bigram_coverage for x in diagnostics) / len(diagnostics) if diagnostics else 0.0),
        structural_contract_pass_rate=(sum(1 for x in diagnostics if x.structural_contract_ok) / len(diagnostics) if diagnostics else 0.0),
        direct_replay=direct,
        full_runtime=full,
    )


def compile_and_benchmark_generalization_ladder(
    backend: CausalTeacherBackend,
    train_cases: Iterable[LadderTrainCase],
    eval_cases: Iterable[LadderEvalCase],
    out_dir: str | Path,
    source_id: str,
    source_version: str,
    *,
    probe_cfg: BehaviourProbeConfig | None = None,
    cognition_policy: RuntimeCognitionPolicy | None = None,
    runtime_top_k: int | None = None,
    pack_id: str = "equinox-generalization-ladder",
    pack_version: int = 20,
    include_pack_metadata: bool = False,
    strict_contracts: bool = True,
    support_documents: Iterable[ForgeDocument] | None = None,
) -> GeneralizationLadderReport:
    cfg = probe_cfg or BehaviourProbeConfig()
    cfg.validate()
    policy = cognition_policy or RuntimeCognitionPolicy(max_predictions=max(int(runtime_top_k or cfg.top_k), 8))
    policy.validate()
    runtime_top_k = int(runtime_top_k or cfg.top_k)

    train_rows = [LadderTrainCase(_norm(x.prompt), _norm(x.family)) for x in train_cases if _norm(x.prompt)]
    eval_rows = [LadderEvalCase(_norm(x.prompt), int(x.level), _norm(x.family), str(x.note or "")) for x in eval_cases if _norm(x.prompt)]
    if not train_rows or not eval_rows:
        raise ValueError("generalization ladder requires non-empty train and evaluation cases")
    if len({x.prompt for x in train_rows}) != len(train_rows):
        raise ValueError("duplicate normalized train prompt")
    if len({(x.prompt, x.level) for x in eval_rows}) != len(eval_rows):
        raise ValueError("duplicate normalized evaluation prompt/level")

    train_prompts = {x.prompt for x in train_rows}
    train_tokens = {t for x in train_rows for t in _tokenize_runtime(x.prompt)}
    train_bigrams = set().union(*(_bigrams(_tokenize_runtime(x.prompt)) for x in train_rows)) if train_rows else set()
    train_families = {x.family for x in train_rows if x.family}

    contract_cache: dict[tuple[str, int], tuple[bool, list[str], dict[str, Any]]] = {}
    for case in eval_rows:
        contract_cache[(case.prompt, case.level)] = _validate_eval_contract(case, train_prompts, train_tokens, train_bigrams, train_families)
    bad = [(c, contract_cache[(c.prompt, c.level)][1]) for c in eval_rows if not contract_cache[(c.prompt, c.level)][0]]
    if strict_contracts and bad:
        msg = "; ".join(f"L{c.level} {c.prompt!r}: {', '.join(v)}" for c, v in bad[:8])
        raise ValueError("generalization ladder structural contract violation: " + msg)

    out = Path(out_dir)
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    eval_dir = out / "evaluation_only"
    eval_dir.mkdir(parents=True, exist_ok=True)
    db_path = out / "train.sqlite"
    pack_path = out / "train.eqbase"

    train_prompt_list = [x.prompt for x in train_rows]
    eval_prompt_list = [x.prompt for x in eval_rows]
    t0 = time.perf_counter()
    train_doc, train_probe_report, train_probes = extract_teacher_behaviour(backend, train_prompt_list, source_id, source_version, cfg)
    eval_doc, eval_probe_report, eval_probes = extract_teacher_behaviour(backend, eval_prompt_list, source_id + ":ladder-eval-reference", source_version, cfg)
    teacher_ms = (time.perf_counter() - t0) * 1000.0
    for label, doc in (("train", train_doc), ("eval", eval_doc)):
        errors = [x for x in validate_document(doc) if x.severity == "error"]
        if errors: raise ValueError(f"{label} teacher IR failed validation: " + "; ".join(x.message for x in errors))
    save_document(train_doc, out / "train-teacher-behaviour.eqir.json")
    save_document(eval_doc, eval_dir / "ladder-teacher-reference.eqir.json")

    relation_policy = CompilePolicy(min_score=0.0, min_confidence=0.0, min_sources=1)
    support_docs = list(support_documents or [])
    for i, doc in enumerate(support_docs, 1):
        errors = [x for x in validate_document(doc) if x.severity == "error"]
        if errors:
            raise ValueError(f"support document {i} failed validation: " + "; ".join(x.message for x in errors))
        save_document(doc, out / f"support-{i:02d}.eqir.json")
    with CanonicalDB(db_path) as db:
        for node in train_doc.nodes: db.upsert_node(node)
        for ev in train_doc.evidence: db.add_evidence(ev)
        for doc in support_docs:
            for node in doc.nodes: db.upsert_node(node)
            for ev in doc.evidence: db.add_evidence(ev)
        db.commit()
        compression = _pass_through_compression(db.db)
        manifest = compile_base_pack(
            db.db, pack_path, pack_id=pack_id, pack_version=pack_version,
            compression_run_id=compression.compression_run_id,
            include_metadata=include_pack_metadata, persist=True, verify_roundtrip=True,
            relation_policy=relation_policy,
        )

    t1 = time.perf_counter()
    eval_direct, _ = benchmark_teacher_probes_against_pack(
        pack_path, eval_probes, cfg, cognition_policy=_direct_replay_policy(policy, runtime_top_k),
        teacher_top_k=cfg.top_k, runtime_top_k=runtime_top_k,
    )
    eval_full, _ = benchmark_teacher_probes_against_pack(
        pack_path, eval_probes, cfg, cognition_policy=policy,
        teacher_top_k=cfg.top_k, runtime_top_k=runtime_top_k,
    )
    runtime_ms = (time.perf_counter() - t1) * 1000.0
    direct_by_prompt = {x.prompt: x for x in eval_direct}
    full_by_prompt = {x.prompt: x for x in eval_full}

    diagnostics: list[LadderStructuralDiagnostics] = []
    detail: list[dict[str, Any]] = []
    for case in eval_rows:
        ok, violations, meta = contract_cache[(case.prompt, case.level)]
        surfaces, width, resolution_kind = _resolution_diagnostics(pack_path, case.prompt, policy)
        tokens = meta.get("tokens", _tokenize_runtime(case.prompt))
        pairs = meta.get("pairs", _bigrams(tokens))
        known_pairs = meta.get("known_pairs", pairs & train_bigrams)
        diag = LadderStructuralDiagnostics(
            prompt=case.prompt,
            requested_level=case.level,
            level_name=LEVELS[case.level],
            exact_in_train=bool(meta.get("exact", case.prompt in train_prompts)),
            proper_train_suffix=meta.get("suffix"),
            prompt_tokens=tokens,
            known_tokens=int(meta.get("known", sum(1 for t in tokens if t in train_tokens))),
            novel_tokens=int(meta.get("novel", sum(1 for t in tokens if t not in train_tokens))),
            token_coverage=(sum(1 for t in tokens if t in train_tokens) / len(tokens) if tokens else 0.0),
            bigrams=len(pairs),
            known_bigrams=len(known_pairs),
            bigram_coverage=(len(known_pairs) / len(pairs) if pairs else 0.0),
            family_seen_in_train=bool(meta.get("family_seen", case.family in train_families if case.family else False)),
            resolved_context_surfaces=surfaces,
            resolved_context_width=width,
            runtime_resolution_kind=resolution_kind,
            structural_contract_ok=ok,
            contract_violations=violations,
        )
        diagnostics.append(diag)
        full = full_by_prompt[case.prompt]
        direct = direct_by_prompt[case.prompt]
        detail.append({
            "prompt": case.prompt, "level": case.level, "level_name": LEVELS[case.level],
            "family": case.family, "note": case.note,
            "structural": diag.to_dict(), "direct_replay": direct.to_dict(), "full_runtime": full.to_dict(),
        })

    level_reports: list[LadderLevelReport] = []
    for level in sorted(LEVELS):
        rows = [c for c in eval_rows if c.level == level]
        if not rows: continue
        full = [full_by_prompt[x.prompt] for x in rows]
        direct = [direct_by_prompt[x.prompt] for x in rows]
        diags = [x for x in diagnostics if x.requested_level == level]
        level_reports.append(_level_report(level, full, direct, diags))

    any_success = [r.level for r in level_reports if r.top1_retention > 0.0]
    full_success = [r.level for r in level_reports if r.top1_retention >= 1.0 - 1e-12]
    contiguous_full = None
    reports_by_level = {r.level: r for r in level_reports}
    for level in range(0, 6):
        row = reports_by_level.get(level)
        if row is None or row.top1_retention < 1.0 - 1e-12:
            break
        contiguous_full = level
    train_manifest_rows = [{"prompt": x.prompt, "family": x.family} for x in train_rows]
    eval_manifest_rows = [{"prompt": x.prompt, "level": x.level, "family": x.family, "note": x.note} for x in eval_rows]
    manifest_payload = {
        "format": LADDER_FORMAT,
        "version": LADDER_VERSION,
        "train": train_manifest_rows,
        "evaluation": eval_manifest_rows,
        "train_sha256": _hash_rows(train_manifest_rows),
        "evaluation_sha256": _hash_rows(eval_manifest_rows),
        "evaluation_reference_compiled": False,
        "strict_contracts": strict_contracts,
        "level_definitions": LEVELS,
    }
    (out / "ladder-manifest.json").write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = GeneralizationLadderReport(
        format=LADDER_FORMAT, version=LADDER_VERSION,
        evaluation_scope="stratified_generalization_ladder",
        generalization_claim=False,
        train_cases=len(train_rows), evaluation_cases=len(eval_rows),
        heldout_cases=sum(1 for x in eval_rows if x.level > 0),
        exact_replay_controls=sum(1 for x in eval_rows if x.level == 0),
        train_dataset_sha256=_hash_rows(train_manifest_rows), evaluation_dataset_sha256=_hash_rows(eval_manifest_rows),
        pack_bytes=manifest.file_bytes, pack_nodes=manifest.node_count, pack_relations=manifest.relation_count,
        compiled_prediction_evidence=train_probe_report.prediction_evidence,
        level_reports=level_reports, structural_diagnostics=diagnostics, cases_detail=detail,
        highest_level_with_any_top1_success=(max(any_success) if any_success else None),
        highest_level_with_full_top1_success=(max(full_success) if full_success else None),
        highest_contiguous_level_with_full_top1_success=contiguous_full,
        teacher_probe_elapsed_ms=teacher_ms, runtime_eval_elapsed_ms=runtime_ms,
        backend_manifest=dict(backend.manifest() or {}), pack_manifest=manifest.to_dict(),
        relation_compile_policy=asdict(relation_policy),
        notes=[
            "L0 is an exact-replay control and is intentionally allowed to overlap the compile split; L1-L5 are held out as full prompts.",
            "Evaluation teacher probes are written only under evaluation_only and are never inserted into train.sqlite or train.eqbase.",
            "L1 tests reuse of an entire known predictive suffix inside a new outer context.",
            "L2 tests a new composition of known lexical parts while requiring some previously observed local adjacency and forbidding L1 suffix replay.",
            "L3 requires a known lexicon but novel local adjacency/order relative to the compile split.",
            "L4 requires at least one novel lexical entity/token while retaining a previously observed relational frame.",
            "L5 requires an explicitly labeled relation/task family that is wholly absent from the compile split.",
            "The ladder reports observed transfer by stratum; it does not establish broad language generalization or reasoning equivalence to the teacher.",
            f"Auxiliary support documents compiled into the BASE: {len(support_docs)}. They may supply non-predictive semantic evidence but evaluation teacher predictions are never inserted.",
        ],
    )
    (out / "generalization-ladder-report.json").write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def load_ladder_train_jsonl(path: str | Path) -> list[LadderTrainCase]:
    out: list[LadderTrainCase] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"): continue
        row = json.loads(line)
        if not isinstance(row, dict) or not row.get("prompt"):
            raise ValueError(f"{path}:{lineno}: expected object with prompt")
        out.append(LadderTrainCase(str(row["prompt"]), str(row.get("family", ""))))
    return out


def load_ladder_eval_jsonl(path: str | Path) -> list[LadderEvalCase]:
    out: list[LadderEvalCase] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"): continue
        row = json.loads(line)
        if not isinstance(row, dict) or not row.get("prompt") or "level" not in row:
            raise ValueError(f"{path}:{lineno}: expected object with prompt and level")
        out.append(LadderEvalCase(str(row["prompt"]), int(row["level"]), str(row.get("family", "")), str(row.get("note", ""))))
    return out
