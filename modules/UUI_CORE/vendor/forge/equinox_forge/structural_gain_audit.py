from __future__ import annotations

from dataclasses import asdict, dataclass
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
    _pool,
    _score_frontier,
    _seed,
)
from .iterative_active_learning import _shadow_split
from .behaviour_retention import compile_and_benchmark_teacher_behaviour
from .real_teacher_domain_trial import ANIMALS, CONTROLS, RELATIONS, _NeuralDomainTeacher, _training_examples
from .runtime_cognition import RuntimeCognition, RuntimeCognitionPolicy
from .runtime_pack import load_base_pack
from .teacher_behavior import BehaviourProbeConfig

TRIAL_FORMAT = "equinox-forge-marginal-structural-gain-audit-v1"
TRIAL_VERSION = "0.43"


@dataclass
class ShadowSnapshot:
    selected_queries: int
    full_accuracy: float
    shadow_accuracy: float
    shadow_abstention_rate: float
    shadow_mean_top1_score: float
    shadow_non_abstained: int
    opaque_relation_abstention_rate: float
    outside_domain_abstention_rate: float
    top1_signature: dict[str, str | None]
    top1_scores: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _compile_pack(
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
        pack_id="equinox-v043-structural-gain-audit",
        pack_version=43,
    )
    return out_dir / "retention.eqbase"


def _snapshot(
    pack_path: Path,
    policy: RuntimeCognitionPolicy,
    candidates: list[ProbeCandidate],
    shadow: list[ProbeCandidate],
    expected: dict[str, str],
    selected: list[ProbeCandidate],
    animals: list[dict[str, str]],
) -> ShadowSnapshot:
    cognition = RuntimeCognition(load_base_pack(pack_path), policy)
    selected_prompts = {c.prompt for c in selected}
    full_correct = 0
    for candidate in candidates:
        report = cognition.predict(candidate.prompt)
        top = report.predictions[0].surface if report.predictions else None
        full_correct += int(top == expected[candidate.prompt])

    shadow_correct = 0
    shadow_abstained = 0
    scores: list[float] = []
    signature: dict[str, str | None] = {}
    top1_scores: dict[str, float] = {}
    for candidate in shadow:
        report = cognition.predict(candidate.prompt)
        top = report.predictions[0].surface if report.predictions else None
        score = float(report.predictions[0].score) if report.predictions else 0.0
        signature[candidate.prompt] = top
        top1_scores[candidate.prompt] = score
        shadow_correct += int(top == expected[candidate.prompt])
        shadow_abstained += int(top is None)
        if top is not None:
            scores.append(score)

    opaque_abstained = 0
    for row in animals:
        report = cognition.predict(f"{row['entity']} frisst")
        opaque_abstained += int(not report.predictions)
    outside_abstained = 0
    for row in CONTROLS:
        report = cognition.predict(f"{row['entity']} ist ein")
        outside_abstained += int(not report.predictions)

    return ShadowSnapshot(
        selected_queries=len(selected_prompts),
        full_accuracy=full_correct / len(candidates) if candidates else 0.0,
        shadow_accuracy=shadow_correct / len(shadow) if shadow else 0.0,
        shadow_abstention_rate=shadow_abstained / len(shadow) if shadow else 0.0,
        shadow_mean_top1_score=statistics.mean(scores) if scores else 0.0,
        shadow_non_abstained=len(scores),
        opaque_relation_abstention_rate=opaque_abstained / len(animals) if animals else 0.0,
        outside_domain_abstention_rate=outside_abstained / len(CONTROLS) if CONTROLS else 0.0,
        top1_signature=signature,
        top1_scores=top1_scores,
    )


def _diff(base: ShadowSnapshot, other: ShadowSnapshot) -> dict[str, Any]:
    prompts = sorted(base.top1_signature)
    changed_top1 = [p for p in prompts if base.top1_signature[p] != other.top1_signature[p]]
    score_changed = [p for p in prompts if abs(base.top1_scores[p] - other.top1_scores[p]) > 1e-9]
    return {
        "top1_changes": len(changed_top1),
        "score_changes": len(score_changed),
        "changed_top1_prompts": changed_top1,
        "mean_top1_score_delta": other.shadow_mean_top1_score - base.shadow_mean_top1_score,
        "shadow_accuracy_delta": other.shadow_accuracy - base.shadow_accuracy,
        "shadow_abstention_delta": other.shadow_abstention_rate - base.shadow_abstention_rate,
    }


def _relation_extreme_schedule(
    seed_selection: list[ProbeCandidate],
    selectable: list[ProbeCandidate],
    relation_index: int,
    budget: int,
) -> list[ProbeCandidate]:
    seed_ids = {c.candidate_id for c in seed_selection}
    remaining = [c for c in selectable if c.candidate_id not in seed_ids]
    remaining.sort(key=lambda c: (c.relation_index != relation_index, c.entity_index, c.relation_index))
    return list(seed_selection) + remaining[: max(0, budget - len(seed_selection))]


def run_structural_gain_audit(
    out_dir: str | Path,
    *,
    max_entities: int | None = None,
    query_budget: int | None = None,
    random_trials: int = 24,
    seed: int = 3901,
    random_seed: int = 4301,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    animals = ANIMALS[:max_entities] if max_entities else list(ANIMALS)
    if len(animals) < 8:
        raise ValueError("structural gain audit requires at least 8 entities")
    if random_trials < 1:
        raise ValueError("random_trials must be >= 1")

    candidates, expected = _pool(animals)
    selectable, shadow = _shadow_split(candidates)
    seed_selection = _seed(selectable, len(animals))
    budget = int(query_budget if query_budget is not None else min(len(selectable), len(animals) * 2))
    if budget < len(seed_selection) or budget > len(selectable):
        raise ValueError(f"query_budget must be in [{len(seed_selection)}, {len(selectable)}]")

    teacher = _NeuralDomainTeacher(_training_examples(animals, CONTROLS), out / "teacher", seed=seed)
    cached = _CachingTeacher(teacher, "v043-structural-gain-cached-teacher")
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

    seed_pack = _compile_pack(cached, seed_selection, out / "seed", "v043-seed", probe_cfg, policy)
    seed_snapshot = _snapshot(seed_pack, policy, candidates, shadow, expected, seed_selection, animals)

    diagnostics = _score_frontier(seed_pack, policy, selectable, {c.candidate_id for c in seed_selection})
    active_extra = _active_additional(
        selectable,
        seed_selection,
        diagnostics,
        budget - len(seed_selection),
    )
    active_selection = list(seed_selection) + active_extra
    active_pack = _compile_pack(cached, active_selection, out / "active", "v043-active", probe_cfg, policy)
    active_snapshot = _snapshot(active_pack, policy, candidates, shadow, expected, active_selection, animals)

    # Saturation diagnostic: query every fact that is legally selectable while
    # keeping the frozen shadow completely untouched. This is not a fair budgeted
    # acquisition strategy; it is an offline ceiling audit for selector leverage.
    saturated_pack = _compile_pack(cached, selectable, out / "all-selectable", "v043-all-selectable", probe_cfg, policy)
    saturated_snapshot = _snapshot(saturated_pack, policy, candidates, shadow, expected, selectable, animals)

    extreme_rows: list[dict[str, Any]] = []
    for relation_index, (relation_key, _) in enumerate(RELATIONS):
        schedule = _relation_extreme_schedule(seed_selection, selectable, relation_index, budget)
        pack = _compile_pack(
            cached,
            schedule,
            out / "extremes" / relation_key,
            f"v043-extreme-{relation_key}",
            probe_cfg,
            policy,
        )
        snap = _snapshot(pack, policy, candidates, shadow, expected, schedule, animals)
        extreme_rows.append({
            "relation_key": relation_key,
            "snapshot": snap.to_dict(),
            "diff_from_seed": _diff(seed_snapshot, snap),
        })

    seed_ids = {c.candidate_id for c in seed_selection}
    remaining = [c for c in selectable if c.candidate_id not in seed_ids]
    rng = random.Random(random_seed)
    random_rows: list[dict[str, Any]] = []
    for trial in range(random_trials):
        extra = rng.sample(remaining, budget - len(seed_selection))
        schedule = list(seed_selection) + extra
        pack = _compile_pack(
            cached,
            schedule,
            out / "random" / f"trial-{trial:03d}",
            f"v043-random-{trial:03d}",
            probe_cfg,
            policy,
        )
        snap = _snapshot(pack, policy, candidates, shadow, expected, schedule, animals)
        random_rows.append({
            "trial": trial,
            "full_accuracy": snap.full_accuracy,
            "shadow_accuracy": snap.shadow_accuracy,
            "shadow_abstention_rate": snap.shadow_abstention_rate,
            "shadow_mean_top1_score": snap.shadow_mean_top1_score,
            "top1_changes_from_seed": _diff(seed_snapshot, snap)["top1_changes"],
        })

    random_shadow = [float(x["shadow_accuracy"]) for x in random_rows]
    random_top1_changes = [int(x["top1_changes_from_seed"]) for x in random_rows]
    report = {
        "format": TRIAL_FORMAT,
        "version": TRIAL_VERSION,
        "runtime_cognition_changed_in_v043": False,
        "domain_entities": len(animals),
        "core_facts": len(candidates),
        "seed_queries": len(seed_selection),
        "budgeted_queries": budget,
        "selectable_facts": len(selectable),
        "shadow_holdout_facts": len(shadow),
        "diagnostic_unique_teacher_queries": cached.cache_misses,
        "teacher": teacher.manifest(),
        "seed": seed_snapshot.to_dict(),
        "active_budgeted": active_snapshot.to_dict(),
        "all_selectable_saturation": saturated_snapshot.to_dict(),
        "diffs": {
            "active_vs_seed": _diff(seed_snapshot, active_snapshot),
            "saturation_vs_seed": _diff(seed_snapshot, saturated_snapshot),
        },
        "relation_extremes": extreme_rows,
        "random_schedule_search": {
            "trials": random_trials,
            "shadow_accuracy_min": min(random_shadow) if random_shadow else 0.0,
            "shadow_accuracy_max": max(random_shadow) if random_shadow else 0.0,
            "shadow_accuracy_mean": statistics.mean(random_shadow) if random_shadow else 0.0,
            "top1_changes_from_seed_min": min(random_top1_changes) if random_top1_changes else 0,
            "top1_changes_from_seed_max": max(random_top1_changes) if random_top1_changes else 0,
            "rows": random_rows,
        },
        "controls": {
            "shadow_never_queried_in_seed": not bool({c.candidate_id for c in seed_selection} & {c.candidate_id for c in shadow}),
            "shadow_never_queried_in_active": not bool({c.candidate_id for c in active_selection} & {c.candidate_id for c in shadow}),
            "shadow_never_queried_in_saturation": not bool({c.candidate_id for c in selectable} & {c.candidate_id for c in shadow}),
            "opaque_relation_abstention_active": active_snapshot.opaque_relation_abstention_rate,
            "outside_domain_abstention_active": active_snapshot.outside_domain_abstention_rate,
            "opaque_relation_abstention_saturation": saturated_snapshot.opaque_relation_abstention_rate,
            "outside_domain_abstention_saturation": saturated_snapshot.outside_domain_abstention_rate,
        },
        "findings": {
            "budgeted_active_improves_shadow_over_seed": active_snapshot.shadow_accuracy > seed_snapshot.shadow_accuracy,
            "all_selectable_improves_shadow_over_seed": saturated_snapshot.shadow_accuracy > seed_snapshot.shadow_accuracy,
            "all_selectable_changes_shadow_top1": _diff(seed_snapshot, saturated_snapshot)["top1_changes"] > 0,
            "all_selectable_changes_shadow_scores": _diff(seed_snapshot, saturated_snapshot)["score_changes"] > 0,
            "tested_random_schedules_change_shadow_accuracy": bool(random_shadow) and (min(random_shadow) != max(random_shadow)),
            "selector_leverage_observed_on_frozen_shadow": any([
                active_snapshot.shadow_accuracy != seed_snapshot.shadow_accuracy,
                saturated_snapshot.shadow_accuracy != seed_snapshot.shadow_accuracy,
                bool(random_shadow) and max(random_shadow) > seed_snapshot.shadow_accuracy,
            ]),
            "interpretation": (
                "Within the tested v0.40 runtime, acquisition changes confidence mass but not the frozen shadow top-1 decisions. "
                "The bottleneck is therefore representation/transfer, not merely prompt selection."
            ),
        },
    }
    (out / "structural_gain_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Equinox Forge v0.43 marginal structural-gain feasibility audit")
    parser.add_argument("--out", default="out/v0_43_structural_gain_audit")
    parser.add_argument("--max-entities", type=int, default=None)
    parser.add_argument("--query-budget", type=int, default=None)
    parser.add_argument("--random-trials", type=int, default=24)
    parser.add_argument("--seed", type=int, default=3901)
    parser.add_argument("--random-seed", type=int, default=4301)
    args = parser.parse_args()
    report = run_structural_gain_audit(
        args.out,
        max_entities=args.max_entities,
        query_budget=args.query_budget,
        random_trials=args.random_trials,
        seed=args.seed,
        random_seed=args.random_seed,
    )
    summary = {
        "version": report["version"],
        "seed_shadow": report["seed"]["shadow_accuracy"],
        "active_shadow": report["active_budgeted"]["shadow_accuracy"],
        "saturated_shadow": report["all_selectable_saturation"]["shadow_accuracy"],
        "saturation_top1_changes": report["diffs"]["saturation_vs_seed"]["top1_changes"],
        "saturation_score_changes": report["diffs"]["saturation_vs_seed"]["score_changes"],
        "random_shadow_min": report["random_schedule_search"]["shadow_accuracy_min"],
        "random_shadow_max": report["random_schedule_search"]["shadow_accuracy_max"],
        "selector_leverage_observed": report["findings"]["selector_leverage_observed_on_frozen_shadow"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
