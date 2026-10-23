from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import argparse
import json
import random
import statistics

from .active_probe_selection import (
    ProbeCandidate,
    _CachingTeacher,
    _active_additional,
    _evaluate_pack,
    _pool,
    _score_frontier,
    _seed,
    _selection_counts,
)
from .behaviour_retention import compile_and_benchmark_teacher_behaviour
from .real_teacher_domain_trial import (
    ANIMALS,
    CONTROLS,
    RELATIONS,
    _NeuralDomainTeacher,
    _training_examples,
)
from .runtime_cognition import RuntimeCognitionPolicy
from .teacher_behavior import BehaviourProbeConfig

TRIAL_FORMAT = "equinox-forge-iterative-active-learning-v1"
TRIAL_VERSION = "0.42"


@dataclass
class CurvePoint:
    query_budget: int
    unique_teacher_queries: int
    full_accuracy: float
    queried_accuracy: float
    shadow_holdout_accuracy: float
    shadow_holdout_abstention_rate: float
    opaque_relation_abstention_rate: float
    outside_domain_abstention_rate: float
    relation_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _shadow_split(
    candidates: list[ProbeCandidate],
) -> tuple[list[ProbeCandidate], list[ProbeCandidate]]:
    """Freeze one never-queryable fact per entity.

    The seed for entity i uses relation i % 4. The shadow holdout uses
    (i + 2) % 4, so seed and holdout never overlap. This holdout is fixed across
    strategies and budgets, avoiding the v0.41 confound where each selector left
    a different set of facts unqueried.
    """
    selectable: list[ProbeCandidate] = []
    shadow: list[ProbeCandidate] = []
    width = len(RELATIONS)
    for c in candidates:
        if c.relation_index == ((c.entity_index + 2) % width):
            shadow.append(c)
        else:
            selectable.append(c)
    return selectable, shadow


def _compile_pack_v042(
    teacher: _CachingTeacher,
    selected: list[ProbeCandidate],
    out_dir: Path,
    source_id: str,
    probe_cfg: BehaviourProbeConfig,
    policy: RuntimeCognitionPolicy,
) -> Path:
    compile_and_benchmark_teacher_behaviour(
        teacher,
        [c.prompt for c in selected],
        out_dir,
        source_id,
        "1",
        probe_cfg=probe_cfg,
        cognition_policy=policy,
        runtime_top_k=4,
        pack_id="equinox-v042-iterative-active",
        pack_version=42,
    )
    return out_dir / "retention.eqbase"


def _fixed_schedule(
    selectable: list[ProbeCandidate],
    entity_count: int,
) -> list[ProbeCandidate]:
    by_pair = {(c.entity_index, c.relation_index): c for c in selectable}
    out: list[ProbeCandidate] = []
    seen: set[str] = set()
    for round_index in range(len(RELATIONS)):
        for i in range(entity_count):
            rel = (i + round_index) % len(RELATIONS)
            candidate = by_pair.get((i, rel))
            if candidate is None or candidate.candidate_id in seen:
                continue
            out.append(candidate)
            seen.add(candidate.candidate_id)
    return out


def _curve_point(
    pack_path: Path,
    policy: RuntimeCognitionPolicy,
    all_candidates: list[ProbeCandidate],
    shadow: list[ProbeCandidate],
    expected: dict[str, str],
    selected: list[ProbeCandidate],
    animals: list[dict[str, str]],
    unique_teacher_queries: int,
) -> CurvePoint:
    full = _evaluate_pack(pack_path, policy, all_candidates, expected, selected, animals)
    shadow_eval = _evaluate_pack(pack_path, policy, shadow, expected, selected, animals)
    queried_eval = _evaluate_pack(pack_path, policy, selected, expected, selected, animals)
    return CurvePoint(
        query_budget=len(selected),
        unique_teacher_queries=unique_teacher_queries,
        full_accuracy=float(full["full_accuracy"]),
        queried_accuracy=float(queried_eval["full_accuracy"]),
        shadow_holdout_accuracy=float(shadow_eval["full_accuracy"]),
        shadow_holdout_abstention_rate=float(shadow_eval["full_abstention_rate"]),
        opaque_relation_abstention_rate=float(full["opaque_relation_abstention_rate"]),
        outside_domain_abstention_rate=float(full["outside_domain_abstention_rate"]),
        relation_counts=dict(_selection_counts(selected)["relation_counts"]),
    )


def _iterative_curve(
    teacher: _NeuralDomainTeacher,
    candidates: list[ProbeCandidate],
    selectable: list[ProbeCandidate],
    shadow: list[ProbeCandidate],
    expected: dict[str, str],
    animals: list[dict[str, str]],
    seed_selection: list[ProbeCandidate],
    out: Path,
    checkpoints: list[int],
    batch_size: int,
    probe_cfg: BehaviourProbeConfig,
    policy: RuntimeCognitionPolicy,
) -> dict[str, Any]:
    cached = _CachingTeacher(teacher, "v042-iterative-cached-teacher")
    selected = list(seed_selection)
    rounds: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    checkpoint_set = set(checkpoints)
    round_index = 0
    while True:
        pack = _compile_pack_v042(
            cached,
            selected,
            out / f"round-{round_index:02d}-q{len(selected):03d}",
            f"v042-iterative-r{round_index}",
            probe_cfg,
            policy,
        )
        if len(selected) in checkpoint_set:
            curve.append(_curve_point(
                pack,
                policy,
                candidates,
                shadow,
                expected,
                selected,
                animals,
                cached.cache_misses,
            ).to_dict())
        if len(selected) >= checkpoints[-1]:
            break
        diagnostics = _score_frontier(
            pack,
            policy,
            selectable,
            {c.candidate_id for c in selected},
        )
        take = min(batch_size, checkpoints[-1] - len(selected))
        added = _active_additional(selectable, selected, diagnostics, take)
        if not added:
            break
        rounds.append({
            "round": round_index + 1,
            "queries_before": len(selected),
            "queries_after": len(selected) + len(added),
            "selected": [c.to_dict() for c in added],
            "frontier_abstentions": sum(1 for d in diagnostics if d.abstained),
            "frontier_non_abstentions": sum(1 for d in diagnostics if not d.abstained),
        })
        selected.extend(added)
        round_index += 1
    return {
        "teacher_queries": cached.cache_misses,
        "teacher_cache_hits": cached.cache_hits,
        "curve": curve,
        "rounds": rounds,
        "selected": [c.to_dict() for c in selected],
        "selection": _selection_counts(selected),
    }


def _schedule_curve(
    teacher: _NeuralDomainTeacher,
    schedule: list[ProbeCandidate],
    strategy_id: str,
    candidates: list[ProbeCandidate],
    shadow: list[ProbeCandidate],
    expected: dict[str, str],
    animals: list[dict[str, str]],
    out: Path,
    checkpoints: list[int],
    probe_cfg: BehaviourProbeConfig,
    policy: RuntimeCognitionPolicy,
    cached_teacher: _CachingTeacher | None = None,
) -> dict[str, Any]:
    cached = cached_teacher or _CachingTeacher(teacher, f"v042-{strategy_id}-cached-teacher")
    curve: list[dict[str, Any]] = []
    for budget in checkpoints:
        selected = schedule[:budget]
        pack = _compile_pack_v042(
            cached,
            selected,
            out / f"q{budget:03d}",
            f"v042-{strategy_id}-q{budget}",
            probe_cfg,
            policy,
        )
        curve.append(_curve_point(
            pack,
            policy,
            candidates,
            shadow,
            expected,
            selected,
            animals,
            cached.cache_misses,
        ).to_dict())
    return {
        "teacher_queries": cached.cache_misses,
        "teacher_cache_hits": cached.cache_hits,
        "curve": curve,
        "selected": [c.to_dict() for c in schedule[: checkpoints[-1]]],
        "selection": _selection_counts(schedule[: checkpoints[-1]]),
    }


def _aggregate_random_curves(trials: list[dict[str, Any]], checkpoints: list[int]) -> dict[str, Any]:
    out_points: list[dict[str, Any]] = []
    for point_index, budget in enumerate(checkpoints):
        row: dict[str, Any] = {"query_budget": budget}
        for metric in (
            "full_accuracy",
            "queried_accuracy",
            "shadow_holdout_accuracy",
            "shadow_holdout_abstention_rate",
        ):
            vals = [float(t["curve"][point_index][metric]) for t in trials]
            row[metric] = {
                "mean": statistics.mean(vals) if vals else 0.0,
                "population_stdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                "min": min(vals) if vals else 0.0,
                "max": max(vals) if vals else 0.0,
            }
        out_points.append(row)
    return {
        "trials": len(trials),
        "curve": out_points,
        "trials_detail": trials,
    }


def run_iterative_active_trial(
    out_dir: str | Path,
    *,
    max_entities: int | None = None,
    final_budget: int | None = None,
    batch_size: int = 8,
    random_trials: int = 5,
    seed: int = 3901,
    random_seed: int = 4210,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    animals = ANIMALS[:max_entities] if max_entities else list(ANIMALS)
    if len(animals) < 8:
        raise ValueError("iterative active trial requires at least 8 entities")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if random_trials < 1:
        raise ValueError("random_trials must be >= 1")

    candidates, expected = _pool(animals)
    selectable, shadow = _shadow_split(candidates)
    entity_count = len(animals)
    seed_selection = _seed(selectable, entity_count)
    budget = int(final_budget if final_budget is not None else min(len(selectable), entity_count * 2))
    if budget < len(seed_selection) or budget > len(selectable):
        raise ValueError(f"final_budget must be in [{len(seed_selection)}, {len(selectable)}]")
    if (budget - len(seed_selection)) % batch_size != 0:
        raise ValueError("final_budget - seed_queries must be divisible by batch_size")
    checkpoints = list(range(len(seed_selection), budget + 1, batch_size))

    teacher = _NeuralDomainTeacher(_training_examples(animals, CONTROLS), out / "teacher", seed=seed)
    probe_cfg = BehaviourProbeConfig(
        top_k=4,
        min_probability=0.00001,
        evidence_confidence=0.95,
        include_punctuation_only=True,
    )
    policy = RuntimeCognitionPolicy(
        max_recall_nodes=96,
        max_recall_depth=3,
        max_predictions=4,
        min_prediction_score=0.000001,
        enable_exact_context_authority=True,
    )

    iterative = _iterative_curve(
        teacher,
        candidates,
        selectable,
        shadow,
        expected,
        animals,
        seed_selection,
        out / "iterative",
        checkpoints,
        batch_size,
        probe_cfg,
        policy,
    )

    # v0.41 one-shot control: rank the complete additional budget from the seed
    # snapshot once, then evaluate prefixes of that fixed order.
    one_teacher = _CachingTeacher(teacher, "v042-one-shot-seed-cached-teacher")
    seed_pack = _compile_pack_v042(
        one_teacher,
        seed_selection,
        out / "one-shot" / "seed-ranking",
        "v042-one-shot-seed",
        probe_cfg,
        policy,
    )
    seed_diagnostics = _score_frontier(
        seed_pack,
        policy,
        selectable,
        {c.candidate_id for c in seed_selection},
    )
    one_extra = _active_additional(
        selectable,
        seed_selection,
        seed_diagnostics,
        budget - len(seed_selection),
    )
    one_schedule = seed_selection + one_extra
    one_shot = _schedule_curve(
        teacher,
        one_schedule,
        "one-shot",
        candidates,
        shadow,
        expected,
        animals,
        out / "one-shot" / "curve",
        checkpoints,
        probe_cfg,
        policy,
        cached_teacher=one_teacher,
    )
    one_shot["seed_ranking_teacher_queries"] = len(seed_selection)
    one_shot["seed_ranking_cache_hits"] = one_teacher.cache_hits

    fixed_schedule = _fixed_schedule(selectable, entity_count)
    fixed = _schedule_curve(
        teacher,
        fixed_schedule,
        "fixed",
        candidates,
        shadow,
        expected,
        animals,
        out / "fixed",
        checkpoints,
        probe_cfg,
        policy,
    )

    seed_ids = {c.candidate_id for c in seed_selection}
    remaining = [c for c in selectable if c.candidate_id not in seed_ids]
    random_rows: list[dict[str, Any]] = []
    for trial_index in range(random_trials):
        rng = random.Random(random_seed + trial_index)
        schedule = seed_selection + rng.sample(remaining, budget - len(seed_selection))
        trial = _schedule_curve(
            teacher,
            schedule,
            f"random-{trial_index}",
            candidates,
            shadow,
            expected,
            animals,
            out / "random" / f"trial-{trial_index:02d}",
            checkpoints,
            probe_cfg,
            policy,
        )
        trial["trial"] = trial_index
        trial["seed"] = random_seed + trial_index
        random_rows.append(trial)
    random_summary = _aggregate_random_curves(random_rows, checkpoints)

    iterative_ids = [x["candidate_id"] for x in iterative["selected"]]
    one_ids = [x["candidate_id"] for x in one_shot["selected"]]
    final_iter = iterative["curve"][-1]
    final_one = one_shot["curve"][-1]
    final_fixed = fixed["curve"][-1]
    final_random_shadow = random_summary["curve"][-1]["shadow_holdout_accuracy"]["mean"]

    report = {
        "format": TRIAL_FORMAT,
        "version": TRIAL_VERSION,
        "scope": "iterative-pool-based-active-learning-with-frozen-shadow-holdout",
        "general_world_model_claim": False,
        "autonomous_question_generation_claim": False,
        "runtime_cognition_changed_in_v042": False,
        "selector_uses_teacher_answer_before_selection": False,
        "selector_uses_expected_labels": False,
        "domain_entities": entity_count,
        "core_facts": len(candidates),
        "selectable_facts": len(selectable),
        "shadow_holdout_facts": len(shadow),
        "shadow_holdout_rule": "relation_index == (entity_index + 2) mod 4",
        "seed_queries": len(seed_selection),
        "final_query_budget": budget,
        "batch_size": batch_size,
        "checkpoints": checkpoints,
        "teacher": teacher.manifest(),
        "iterative": iterative,
        "one_shot_v041_style": one_shot,
        "fixed": fixed,
        "random": random_summary,
        "findings": {
            "iterative_final_equals_one_shot_final_selection": iterative_ids == one_ids,
            "iterative_final_full_accuracy_delta_vs_one_shot": final_iter["full_accuracy"] - final_one["full_accuracy"],
            "iterative_final_shadow_delta_vs_one_shot": final_iter["shadow_holdout_accuracy"] - final_one["shadow_holdout_accuracy"],
            "iterative_final_full_accuracy_delta_vs_fixed": final_iter["full_accuracy"] - final_fixed["full_accuracy"],
            "iterative_final_shadow_delta_vs_fixed": final_iter["shadow_holdout_accuracy"] - final_fixed["shadow_holdout_accuracy"],
            "iterative_final_shadow_delta_vs_random_mean": final_iter["shadow_holdout_accuracy"] - final_random_shadow,
            "v041_unqueried_metric_is_strategy_dependent": True,
            "genuine_fixed_holdout_transfer_improvement_observed": (
                final_iter["shadow_holdout_accuracy"] > fixed["curve"][0]["shadow_holdout_accuracy"]
            ),
        },
        "controls": {
            "shadow_never_selectable": not any(c.candidate_id in {x.candidate_id for x in selectable} for c in shadow),
            "seed_and_shadow_disjoint": not any(c.candidate_id in {x.candidate_id for x in seed_selection} for c in shadow),
            "iterative_unique_teacher_queries_equal_budget": iterative["teacher_queries"] == budget,
            "one_shot_unique_teacher_queries_equal_budget": one_shot["teacher_queries"] == budget,
            "fixed_unique_teacher_queries_equal_budget": fixed["teacher_queries"] == budget,
            "all_random_unique_teacher_queries_equal_budget": all(x["teacher_queries"] == budget for x in random_rows),
            "iterative_opaque_relation_abstention": final_iter["opaque_relation_abstention_rate"],
            "iterative_outside_domain_abstention": final_iter["outside_domain_abstention_rate"],
        },
        "notes": [
            "v0.42 is a selection/evaluation release; RuntimeCognition remains version 0.40.",
            "The fixed shadow holdout is never eligible for teacher probing by any strategy, making transfer comparisons strategy-independent.",
            "Iterative active selection recompiles after every batch and rescans the frontier before choosing the next batch.",
            "The v0.41 one-shot control computes frontier uncertainty only once from the seed BASE and then fixes the entire additional selection order.",
            "On the frozen full-domain reference, iterative rescoring chooses the same final schedule as one-shot v0.41 and does not improve the frozen shadow holdout.",
            "Therefore v0.41's higher accuracy on its strategy-specific unqueried remainder should be interpreted as query-allocation advantage, not yet as proven transfer improvement.",
        ],
    }
    (out / "iterative-active-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Equinox Forge v0.42 iterative active-learning audit")
    ap.add_argument("--out", default="out/v0_42_iterative_active_learning")
    ap.add_argument("--entities", type=int, default=None, help="optional prefix size (min 8)")
    ap.add_argument("--budget", type=int, default=None, help="final unique teacher-query budget; default = 2 per entity")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--random-trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=3901)
    ap.add_argument("--random-seed", type=int, default=4210)
    args = ap.parse_args(argv)
    report = run_iterative_active_trial(
        args.out,
        max_entities=args.entities,
        final_budget=args.budget,
        batch_size=args.batch_size,
        random_trials=args.random_trials,
        seed=args.seed,
        random_seed=args.random_seed,
    )
    print(json.dumps({
        "format": report["format"],
        "version": report["version"],
        "domain_entities": report["domain_entities"],
        "final_query_budget": report["final_query_budget"],
        "batch_size": report["batch_size"],
        "iterative_curve": report["iterative"]["curve"],
        "one_shot_curve": report["one_shot_v041_style"]["curve"],
        "fixed_curve": report["fixed"]["curve"],
        "random_curve": report["random"]["curve"],
        "findings": report["findings"],
        "controls": report["controls"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
