from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import json
import shutil

from .behaviour_retention import (
    BehaviourRetentionCase,
    _summarize_cases,
    benchmark_teacher_probes_against_pack,
)
from .generalization_ladder import (
    LEVELS,
    LadderEvalCase,
    LadderTrainCase,
    compile_and_benchmark_generalization_ladder,
)
from .io import load_document
from .runtime_cognition import RuntimeCognitionPolicy
from .teacher_behavior import (
    BehaviourProbeConfig,
    CausalTeacherBackend,
    TeacherCandidate,
    TeacherProbe,
)

ATTRIBUTION_FORMAT = "equinox-forge-mechanism-attribution-v1"
ATTRIBUTION_VERSION = "0.21"

# Leave-one-mechanism-out ablations. The full runtime remains the reference point.
CORE_ABLATIONS: dict[str, dict[str, bool]] = {
    "no_suffix_context": {"enable_suffix_context": False},
    "no_lexical_seeds": {"enable_lexical_seeds": False},
    "no_parent_routing": {"enable_parent_routing": False},
    "no_relation_recall": {"enable_relation_recall": False},
    "no_direct_context_prediction": {"enable_direct_context_prediction": False},
    "no_transition_prediction": {"enable_transition_prediction": False},
    "no_recalled_context_prediction": {"enable_recalled_context_prediction": False},
}

COMBINATION_ABLATIONS: dict[str, dict[str, bool]] = {
    "no_suffix_and_parent_routing": {
        "enable_suffix_context": False,
        "enable_parent_routing": False,
    },
    "no_suffix_and_lexical_seeds": {
        "enable_suffix_context": False,
        "enable_lexical_seeds": False,
    },
    "no_structural_or_relation_recall": {
        "enable_parent_routing": False,
        "enable_relation_recall": False,
    },
    "direct_context_only": {
        "enable_lexical_seeds": False,
        "enable_parent_routing": False,
        "enable_relation_recall": False,
        "enable_transition_prediction": False,
        "enable_recalled_context_prediction": False,
    },
    "recall_prediction_only": {
        "enable_direct_context_prediction": False,
        "enable_transition_prediction": False,
    },
}


@dataclass
class AblationLevelDelta:
    level: int
    level_name: str
    cases: int
    baseline_top1: float
    ablated_top1: float
    delta_top1: float
    baseline_ndcg: float
    ablated_ndcg: float
    delta_ndcg: float
    baseline_probability_mass: float
    ablated_probability_mass: float
    delta_probability_mass: float
    baseline_avg_recall_nodes: float
    ablated_avg_recall_nodes: float
    recall_node_delta: float
    baseline_avg_relation_checks: float
    ablated_avg_relation_checks: float
    relation_check_delta: float
    top1_losses: int
    top1_gains: int
    unchanged_top1_successes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MechanismAblationReport:
    name: str
    disabled: dict[str, bool]
    cases: int
    baseline: dict[str, Any]
    ablated: dict[str, Any]
    delta_top1: float
    delta_ndcg: float
    delta_probability_mass: float
    delta_avg_recall_nodes: float
    delta_avg_relation_checks: float
    top1_losses: int
    top1_gains: int
    necessary_case_prompts: list[str]
    gained_case_prompts: list[str]
    level_deltas: list[AblationLevelDelta]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["level_deltas"] = [x.to_dict() for x in self.level_deltas]
        return d


@dataclass
class MechanismAttributionReport:
    format: str
    version: str
    evaluation_scope: str
    causal_claim_scope: str
    pack_path: str
    train_cases: int
    evaluation_cases: int
    baseline: dict[str, Any]
    ablations: list[MechanismAblationReport]
    level_mechanism_matrix: dict[str, dict[str, float]]
    strongest_leave_one_out_mechanism_by_level: dict[str, str | None]
    cases_detail: list[dict[str, Any]]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ablations"] = [x.to_dict() for x in self.ablations]
        return d


def _policy_variant(base: RuntimeCognitionPolicy, overrides: dict[str, bool]) -> RuntimeCognitionPolicy:
    payload = asdict(base)
    payload.update(overrides)
    policy = RuntimeCognitionPolicy(**payload)
    policy.validate()
    return policy


def _teacher_probes_from_ir(path: str | Path) -> list[TeacherProbe]:
    """Reconstruct accepted teacher probes from the persisted evaluation-only IR.

    This avoids re-running the teacher for every ablation. Only metadata written by
    teacher_behavior.extract_teacher_behaviour is used.
    """
    doc = load_document(path)
    grouped: dict[str, list[tuple[int, TeacherCandidate]]] = {}
    for ev in doc.evidence:
        md = dict(ev.metadata or {})
        if md.get("evidence_kind") != "teacher_behaviour":
            continue
        prompt = str(md.get("prompt", "")).strip()
        if not prompt:
            continue
        rank = int(md.get("rank", 10**9))
        cand = TeacherCandidate(
            token_id=int(md.get("teacher_token_id", -1)),
            raw_token=str(md.get("raw_token", "")),
            surface=str(md.get("decoded_surface", md.get("canonical_surface", ""))),
            probability=float(md.get("teacher_probability", ev.score)),
            logit=(None if md.get("teacher_logit") is None else float(md.get("teacher_logit"))),
            special=False,
        )
        grouped.setdefault(prompt, []).append((rank, cand))
    out: list[TeacherProbe] = []
    for prompt in sorted(grouped):
        rows = sorted(grouped[prompt], key=lambda x: (x[0], x[1].token_id, x[1].surface))
        out.append(TeacherProbe(prompt, [c for _, c in rows], {"reconstructed_from_ir": True}))
    return out


def _summary(rows: list[BehaviourRetentionCase]) -> dict[str, Any]:
    s = _summarize_cases(rows)
    n = len(rows)
    s["average_recall_nodes"] = (sum(x.recall_nodes for x in rows) / n if n else 0.0)
    s["average_relation_checks"] = (sum(x.relation_checks for x in rows) / n if n else 0.0)
    return s


def _level_delta(
    level: int,
    baseline_rows: list[BehaviourRetentionCase],
    ablated_rows: list[BehaviourRetentionCase],
) -> AblationLevelDelta:
    base = _summary(baseline_rows)
    abl = _summary(ablated_rows)
    amap = {x.prompt: x for x in ablated_rows}
    losses = gains = unchanged = 0
    for b in baseline_rows:
        a = amap[b.prompt]
        if b.top1_match and not a.top1_match:
            losses += 1
        elif not b.top1_match and a.top1_match:
            gains += 1
        elif b.top1_match and a.top1_match:
            unchanged += 1
    return AblationLevelDelta(
        level=level,
        level_name=LEVELS[level],
        cases=len(baseline_rows),
        baseline_top1=base["top1_retention"],
        ablated_top1=abl["top1_retention"],
        delta_top1=abl["top1_retention"] - base["top1_retention"],
        baseline_ndcg=base["mean_ndcg_at_k"],
        ablated_ndcg=abl["mean_ndcg_at_k"],
        delta_ndcg=abl["mean_ndcg_at_k"] - base["mean_ndcg_at_k"],
        baseline_probability_mass=base["mean_teacher_probability_mass_retained_in_runtime_topk"],
        ablated_probability_mass=abl["mean_teacher_probability_mass_retained_in_runtime_topk"],
        delta_probability_mass=(abl["mean_teacher_probability_mass_retained_in_runtime_topk"] - base["mean_teacher_probability_mass_retained_in_runtime_topk"]),
        baseline_avg_recall_nodes=base["average_recall_nodes"],
        ablated_avg_recall_nodes=abl["average_recall_nodes"],
        recall_node_delta=abl["average_recall_nodes"] - base["average_recall_nodes"],
        baseline_avg_relation_checks=base["average_relation_checks"],
        ablated_avg_relation_checks=abl["average_relation_checks"],
        relation_check_delta=abl["average_relation_checks"] - base["average_relation_checks"],
        top1_losses=losses,
        top1_gains=gains,
        unchanged_top1_successes=unchanged,
    )


def benchmark_mechanism_attribution(
    pack_path: str | Path,
    probes: list[TeacherProbe],
    eval_cases: Iterable[LadderEvalCase],
    probe_cfg: BehaviourProbeConfig,
    *,
    base_policy: RuntimeCognitionPolicy | None = None,
    runtime_top_k: int | None = None,
    include_combinations: bool = True,
) -> MechanismAttributionReport:
    cases = list(eval_cases)
    by_prompt_level = {x.prompt: x.level for x in cases}
    runtime_top_k = int(runtime_top_k or probe_cfg.top_k)
    base_policy = base_policy or RuntimeCognitionPolicy(max_predictions=max(runtime_top_k, 8), min_prediction_score=0.0)
    base_policy.validate()

    baseline_rows, _ = benchmark_teacher_probes_against_pack(
        pack_path, probes, probe_cfg,
        cognition_policy=base_policy,
        teacher_top_k=probe_cfg.top_k,
        runtime_top_k=runtime_top_k,
    )
    baseline_rows = [x for x in baseline_rows if x.prompt in by_prompt_level]
    baseline_map = {x.prompt: x for x in baseline_rows}
    baseline_summary = _summary(baseline_rows)

    definitions = dict(CORE_ABLATIONS)
    if include_combinations:
        definitions.update(COMBINATION_ABLATIONS)

    reports: list[MechanismAblationReport] = []
    detail_map: dict[str, dict[str, Any]] = {
        p: {"prompt": p, "level": by_prompt_level[p], "baseline": baseline_map[p].to_dict(), "ablations": {}}
        for p in baseline_map
    }

    for name, disabled in definitions.items():
        policy = _policy_variant(base_policy, disabled)
        rows, _ = benchmark_teacher_probes_against_pack(
            pack_path, probes, probe_cfg,
            cognition_policy=policy,
            teacher_top_k=probe_cfg.top_k,
            runtime_top_k=runtime_top_k,
        )
        rows = [x for x in rows if x.prompt in by_prompt_level]
        amap = {x.prompt: x for x in rows}
        summary = _summary(rows)
        losses = [p for p, b in baseline_map.items() if b.top1_match and not amap[p].top1_match]
        gains = [p for p, b in baseline_map.items() if not b.top1_match and amap[p].top1_match]
        levels: list[AblationLevelDelta] = []
        for level in sorted(LEVELS):
            bp = [x for x in baseline_rows if by_prompt_level[x.prompt] == level]
            ap = [amap[x.prompt] for x in bp]
            if bp:
                levels.append(_level_delta(level, bp, ap))
        reports.append(MechanismAblationReport(
            name=name,
            disabled=disabled,
            cases=len(rows),
            baseline=baseline_summary,
            ablated=summary,
            delta_top1=summary["top1_retention"] - baseline_summary["top1_retention"],
            delta_ndcg=summary["mean_ndcg_at_k"] - baseline_summary["mean_ndcg_at_k"],
            delta_probability_mass=(summary["mean_teacher_probability_mass_retained_in_runtime_topk"] - baseline_summary["mean_teacher_probability_mass_retained_in_runtime_topk"]),
            delta_avg_recall_nodes=summary["average_recall_nodes"] - baseline_summary["average_recall_nodes"],
            delta_avg_relation_checks=summary["average_relation_checks"] - baseline_summary["average_relation_checks"],
            top1_losses=len(losses),
            top1_gains=len(gains),
            necessary_case_prompts=sorted(losses),
            gained_case_prompts=sorted(gains),
            level_deltas=levels,
        ))
        for p, row in amap.items():
            detail_map[p]["ablations"][name] = row.to_dict()

    core_names = list(CORE_ABLATIONS)
    matrix: dict[str, dict[str, float]] = {}
    strongest: dict[str, str | None] = {}
    for level in sorted(LEVELS):
        key = f"L{level}"
        matrix[key] = {}
        best_name = None
        best_drop = 0.0
        for report in reports:
            if report.name not in core_names:
                continue
            row = next((x for x in report.level_deltas if x.level == level), None)
            if row is None:
                continue
            # Positive attribution value means performance was lost when removed.
            drop = max(0.0, -row.delta_top1)
            matrix[key][report.name] = drop
            if drop > best_drop + 1e-12:
                best_drop = drop
                best_name = report.name
        strongest[key] = best_name

    return MechanismAttributionReport(
        format=ATTRIBUTION_FORMAT,
        version=ATTRIBUTION_VERSION,
        evaluation_scope="leave_one_mechanism_out_runtime_ablation",
        causal_claim_scope="controlled runtime mechanism necessity under a fixed compiled BASE; single ablations do not resolve redundant/shared causality",
        pack_path=str(Path(pack_path)),
        train_cases=0,
        evaluation_cases=len(baseline_rows),
        baseline=baseline_summary,
        ablations=reports,
        level_mechanism_matrix=matrix,
        strongest_leave_one_out_mechanism_by_level=strongest,
        cases_detail=[detail_map[p] for p in sorted(detail_map)],
        notes=[
            "All ablations run against the same immutable .eqbase pack and the same persisted teacher-reference probes.",
            "A positive attribution drop means removing that mechanism harmed the measured metric; it does not prove that mechanism is uniquely sufficient.",
            "If two mechanisms are redundant, neither may show a large single-ablation drop even though removing both can fail; combination ablations diagnose some such interactions.",
            "Runtime work deltas are deterministic Forge work counters, not hardware energy measurements.",
            "Teacher evaluation evidence is not compiled into the BASE.",
        ],
    )


def compile_and_benchmark_mechanism_attribution(
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
    pack_id: str = "equinox-mechanism-attribution",
    pack_version: int = 21,
    include_pack_metadata: bool = False,
    strict_contracts: bool = True,
    include_combinations: bool = True,
) -> MechanismAttributionReport:
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    ladder_dir = out / "ladder"
    train_rows = list(train_cases)
    eval_rows = list(eval_cases)
    cfg = probe_cfg or BehaviourProbeConfig()
    policy = cognition_policy or RuntimeCognitionPolicy(max_predictions=max(int(runtime_top_k or cfg.top_k), 8), min_prediction_score=0.0)

    ladder = compile_and_benchmark_generalization_ladder(
        backend, train_rows, eval_rows, ladder_dir, source_id, source_version,
        probe_cfg=cfg, cognition_policy=policy, runtime_top_k=runtime_top_k,
        pack_id=pack_id, pack_version=pack_version,
        include_pack_metadata=include_pack_metadata, strict_contracts=strict_contracts,
    )
    probes = _teacher_probes_from_ir(ladder_dir / "evaluation_only" / "ladder-teacher-reference.eqir.json")
    report = benchmark_mechanism_attribution(
        ladder_dir / "train.eqbase", probes, eval_rows, cfg,
        base_policy=policy, runtime_top_k=runtime_top_k, include_combinations=include_combinations,
    )
    report.train_cases = ladder.train_cases
    payload = report.to_dict()
    payload["ladder_reference"] = {
        "highest_contiguous_level_with_full_top1_success": ladder.highest_contiguous_level_with_full_top1_success,
        "highest_level_with_any_top1_success": ladder.highest_level_with_any_top1_success,
        "pack_bytes": ladder.pack_bytes,
        "pack_nodes": ladder.pack_nodes,
        "pack_relations": ladder.pack_relations,
    }
    (out / "mechanism-attribution-report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
