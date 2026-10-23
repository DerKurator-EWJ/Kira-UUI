from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import argparse
import json
import random
import statistics

from .behaviour_retention import compile_and_benchmark_teacher_behaviour
from .real_teacher_domain_trial import (
    ANIMALS,
    CONTROLS,
    OPAQUE_RELATION,
    RELATIONS,
    _NeuralDomainTeacher,
    _training_examples,
)
from .runtime_cognition import RuntimeCognition, RuntimeCognitionPolicy
from .runtime_pack import load_base_pack
from .teacher_behavior import BehaviourProbeConfig, TeacherCandidate

TRIAL_FORMAT = "equinox-forge-active-probe-selection-v1"
TRIAL_VERSION = "0.41"


@dataclass(frozen=True)
class ProbeCandidate:
    candidate_id: str
    entity_index: int
    entity: str
    relation_index: int
    relation_key: str
    prompt: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SelectionDiagnostic:
    candidate_id: str
    prompt: str
    relation_key: str
    entity: str
    abstained: bool
    top_score: float
    margin: float
    uncertainty: float
    selected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _CachingTeacher:
    """Cache teacher calls so repeated compilation does not create hidden query cost."""

    def __init__(self, delegate: _NeuralDomainTeacher, backend_id: str):
        self.delegate = delegate
        self.backend_id = backend_id
        self._cache: dict[tuple[str, int, float], list[TeacherCandidate]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def manifest(self) -> dict[str, Any]:
        out = dict(self.delegate.manifest())
        out.update({
            "backend_id": self.backend_id,
            "cached_unique_teacher_queries": self.cache_misses,
            "cache_hits": self.cache_hits,
            "cache_entries": len(self._cache),
        })
        return out

    def next_token(self, prompt: str, top_k: int, temperature: float) -> list[TeacherCandidate]:
        key = (str(prompt), int(top_k), float(temperature))
        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return list(cached)
        self.cache_misses += 1
        value = list(self.delegate.next_token(prompt, top_k, temperature))
        self._cache[key] = value
        return list(value)


def _pool(animals: list[dict[str, str]]) -> tuple[list[ProbeCandidate], dict[str, str]]:
    candidates: list[ProbeCandidate] = []
    expected: dict[str, str] = {}
    for i, row in enumerate(animals):
        for j, (key, template) in enumerate(RELATIONS):
            prompt = template.format(entity=row["entity"])
            candidates.append(ProbeCandidate(
                candidate_id=f"entity-{i:03d}:{key}",
                entity_index=i,
                entity=row["entity"],
                relation_index=j,
                relation_key=key,
                prompt=prompt,
            ))
            expected[prompt] = row[key]
    return candidates, expected


def _seed(candidates: list[ProbeCandidate], entity_count: int) -> list[ProbeCandidate]:
    # One deterministic relation per entity, rotating relation families. This gives
    # every entity one foothold and exposes every relation family before active
    # selection begins. No teacher answer is consulted to construct the seed.
    by_pair = {(c.entity_index, c.relation_index): c for c in candidates}
    return [by_pair[(i, i % len(RELATIONS))] for i in range(entity_count)]


def _fixed_schedule(candidates: list[ProbeCandidate], entity_count: int, budget: int) -> list[ProbeCandidate]:
    by_pair = {(c.entity_index, c.relation_index): c for c in candidates}
    out: list[ProbeCandidate] = []
    for round_index in range(len(RELATIONS)):
        for i in range(entity_count):
            rel = (i + round_index) % len(RELATIONS)
            out.append(by_pair[(i, rel)])
            if len(out) >= budget:
                return out
    return out


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
        pack_id="equinox-v041-active-probe",
        pack_version=41,
    )
    return out_dir / "retention.eqbase"


def _score_frontier(
    pack_path: Path,
    policy: RuntimeCognitionPolicy,
    candidates: list[ProbeCandidate],
    selected_ids: set[str],
) -> list[SelectionDiagnostic]:
    cognition = RuntimeCognition(load_base_pack(pack_path), policy)
    rows: list[SelectionDiagnostic] = []
    for candidate in candidates:
        if candidate.candidate_id in selected_ids:
            continue
        report = cognition.predict(candidate.prompt)
        predictions = report.predictions
        if predictions:
            top = float(predictions[0].score)
            second = float(predictions[1].score) if len(predictions) > 1 else 0.0
            margin = max(0.0, top - second)
            # Proxy expected information gain: uncertain existing predictions are
            # useful, but complete abstention means the current BASE has no answer
            # at all and therefore receives a higher acquisition priority.
            uncertainty = max(0.0, 1.0 - min(1.0, top)) + 0.25 * max(0.0, 1.0 - min(1.0, margin))
            abstained = False
        else:
            top = 0.0
            margin = 0.0
            uncertainty = 2.0
            abstained = True
        rows.append(SelectionDiagnostic(
            candidate.candidate_id,
            candidate.prompt,
            candidate.relation_key,
            candidate.entity,
            abstained,
            top,
            margin,
            uncertainty,
        ))
    return rows


def _active_additional(
    candidates: list[ProbeCandidate],
    seed: list[ProbeCandidate],
    diagnostics: list[SelectionDiagnostic],
    additional: int,
) -> list[ProbeCandidate]:
    if additional <= 0:
        return []
    candidate_by_id = {c.candidate_id: c for c in candidates}
    diag_by_id = {d.candidate_id: d for d in diagnostics}
    relation_counts = {j: 0 for j in range(len(RELATIONS))}
    entity_counts: dict[int, int] = {}
    for c in seed:
        relation_counts[c.relation_index] += 1
        entity_counts[c.entity_index] = entity_counts.get(c.entity_index, 0) + 1
    available = [c for c in candidates if c.candidate_id in diag_by_id]
    chosen: list[ProbeCandidate] = []
    while available and len(chosen) < additional:
        # Utility is computed entirely from the current Equinox pack plus prompt
        # metadata. Expected labels / teacher answers are deliberately unavailable.
        best = max(
            available,
            key=lambda c: (
                diag_by_id[c.candidate_id].uncertainty,
                1.0 / (1.0 + relation_counts[c.relation_index]),
                1.0 / (1.0 + entity_counts.get(c.entity_index, 0)),
                -c.entity_index,
                -c.relation_index,
            ),
        )
        chosen.append(best)
        relation_counts[best.relation_index] += 1
        entity_counts[best.entity_index] = entity_counts.get(best.entity_index, 0) + 1
        diag_by_id[best.candidate_id].selected = True
        available.remove(best)
    return chosen


def _evaluate_pack(
    pack_path: Path,
    policy: RuntimeCognitionPolicy,
    candidates: list[ProbeCandidate],
    expected: dict[str, str],
    selected: list[ProbeCandidate],
    animals: list[dict[str, str]],
) -> dict[str, Any]:
    cognition = RuntimeCognition(load_base_pack(pack_path), policy)
    selected_prompts = {c.prompt for c in selected}
    correct = 0
    abstained = 0
    unqueried_correct = 0
    unqueried_abstained = 0
    unqueried_n = 0
    relation_stats: dict[str, dict[str, int]] = {
        key: {"cases": 0, "correct": 0, "abstained": 0} for key, _ in RELATIONS
    }
    recall_nodes = 0
    relation_checks = 0
    details: list[dict[str, Any]] = []
    for c in candidates:
        report = cognition.predict(c.prompt)
        top = report.predictions[0].surface if report.predictions else None
        ok = top == expected[c.prompt]
        is_abstain = top is None
        correct += int(ok)
        abstained += int(is_abstain)
        if c.prompt not in selected_prompts:
            unqueried_n += 1
            unqueried_correct += int(ok)
            unqueried_abstained += int(is_abstain)
        stats = relation_stats[c.relation_key]
        stats["cases"] += 1
        stats["correct"] += int(ok)
        stats["abstained"] += int(is_abstain)
        recall_nodes += len(report.recall)
        relation_checks += int(report.work.get("relation_checks", 0)) + int(report.work.get("prediction_relation_checks", 0))
        details.append({
            "candidate_id": c.candidate_id,
            "prompt": c.prompt,
            "relation_key": c.relation_key,
            "queried": c.prompt in selected_prompts,
            "expected": expected[c.prompt],
            "runtime_top1": top,
            "correct": ok,
            "abstained": is_abstain,
        })
    n = len(candidates)
    opaque_abstained = 0
    for row in animals:
        _, template = OPAQUE_RELATION
        report = cognition.predict(template.format(entity=row["entity"]))
        opaque_abstained += int(not report.predictions)
    outside_abstained = 0
    for row in CONTROLS:
        report = cognition.predict(f"{row['entity']} ist ein")
        outside_abstained += int(not report.predictions)
    return {
        "cases": n,
        "queried_cases": len(selected_prompts),
        "unqueried_cases": unqueried_n,
        "full_accuracy": correct / n if n else 0.0,
        "full_abstention_rate": abstained / n if n else 0.0,
        "unqueried_accuracy": unqueried_correct / unqueried_n if unqueried_n else 0.0,
        "unqueried_abstention_rate": unqueried_abstained / unqueried_n if unqueried_n else 0.0,
        "opaque_relation_abstention_rate": opaque_abstained / len(animals) if animals else 0.0,
        "outside_domain_abstention_rate": outside_abstained / len(CONTROLS) if CONTROLS else 0.0,
        "average_recall_nodes": recall_nodes / n if n else 0.0,
        "average_relation_checks": relation_checks / n if n else 0.0,
        "by_relation": {
            key: {
                **stats,
                "accuracy": stats["correct"] / stats["cases"] if stats["cases"] else 0.0,
                "abstention_rate": stats["abstained"] / stats["cases"] if stats["cases"] else 0.0,
            }
            for key, stats in relation_stats.items()
        },
        "details": details,
    }


def _selection_counts(selected: list[ProbeCandidate]) -> dict[str, Any]:
    relation_counts = {key: 0 for key, _ in RELATIONS}
    entity_counts: dict[str, int] = {}
    for c in selected:
        relation_counts[c.relation_key] += 1
        entity_counts[c.entity] = entity_counts.get(c.entity, 0) + 1
    return {
        "relation_counts": relation_counts,
        "entities_covered": len(entity_counts),
        "min_queries_per_covered_entity": min(entity_counts.values()) if entity_counts else 0,
        "max_queries_per_covered_entity": max(entity_counts.values()) if entity_counts else 0,
    }


def _aggregate_random(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ["full_accuracy", "unqueried_accuracy", "full_abstention_rate", "unqueried_abstention_rate"]
    out: dict[str, Any] = {"trials": len(rows), "trials_detail": rows}
    for metric in metrics:
        vals = [float(x[metric]) for x in rows]
        out[metric] = {
            "mean": statistics.mean(vals) if vals else 0.0,
            "population_stdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals) if vals else 0.0,
            "max": max(vals) if vals else 0.0,
        }
    return out


def run_active_probe_trial(
    out_dir: str | Path,
    *,
    max_entities: int | None = None,
    query_budget: int | None = None,
    random_trials: int = 5,
    seed: int = 3901,
    random_seed: int = 4110,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    animals = ANIMALS[:max_entities] if max_entities else list(ANIMALS)
    if len(animals) < 8:
        raise ValueError("active probe trial requires at least 8 entities")
    candidates, expected = _pool(animals)
    entity_count = len(animals)
    minimum_budget = entity_count
    budget = int(query_budget if query_budget is not None else min(len(candidates), entity_count * 2))
    if budget < minimum_budget or budget > len(candidates):
        raise ValueError(f"query_budget must be in [{minimum_budget}, {len(candidates)}]")
    if random_trials < 1:
        raise ValueError("random_trials must be >= 1")

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

    seed_selection = _seed(candidates, entity_count)
    active_teacher = _CachingTeacher(teacher, "v041-active-cached-teacher")
    seed_pack = _compile_pack(
        active_teacher,
        seed_selection,
        out / "active" / "seed",
        "v041-active-seed",
        probe_cfg,
        policy,
    )
    diagnostics = _score_frontier(seed_pack, policy, candidates, {c.candidate_id for c in seed_selection})
    active_extra = _active_additional(candidates, seed_selection, diagnostics, budget - len(seed_selection))
    active_selected = seed_selection + active_extra
    active_pack = _compile_pack(
        active_teacher,
        active_selected,
        out / "active" / "final",
        "v041-active-final",
        probe_cfg,
        policy,
    )
    active_eval = _evaluate_pack(active_pack, policy, candidates, expected, active_selected, animals)

    fixed_selected = _fixed_schedule(candidates, entity_count, budget)
    fixed_teacher = _CachingTeacher(teacher, "v041-fixed-cached-teacher")
    fixed_pack = _compile_pack(
        fixed_teacher,
        fixed_selected,
        out / "fixed",
        "v041-fixed",
        probe_cfg,
        policy,
    )
    fixed_eval = _evaluate_pack(fixed_pack, policy, candidates, expected, fixed_selected, animals)

    remaining = [c for c in candidates if c.candidate_id not in {x.candidate_id for x in seed_selection}]
    random_rows: list[dict[str, Any]] = []
    for trial_index in range(random_trials):
        rng = random.Random(random_seed + trial_index)
        extra = rng.sample(remaining, budget - len(seed_selection))
        selected = seed_selection + extra
        cached = _CachingTeacher(teacher, f"v041-random-{trial_index}-cached-teacher")
        pack = _compile_pack(
            cached,
            selected,
            out / "random" / f"trial-{trial_index:02d}",
            f"v041-random-{trial_index}",
            probe_cfg,
            policy,
        )
        ev = _evaluate_pack(pack, policy, candidates, expected, selected, animals)
        random_rows.append({
            "trial": trial_index,
            "seed": random_seed + trial_index,
            "teacher_queries": cached.cache_misses,
            "full_accuracy": ev["full_accuracy"],
            "unqueried_accuracy": ev["unqueried_accuracy"],
            "full_abstention_rate": ev["full_abstention_rate"],
            "unqueried_abstention_rate": ev["unqueried_abstention_rate"],
            "selection": _selection_counts(selected),
        })
    random_summary = _aggregate_random(random_rows)

    selected_diag = [d.to_dict() for d in diagnostics]
    selected_active_ids = {c.candidate_id for c in active_extra}
    for d in selected_diag:
        d["selected"] = d["candidate_id"] in selected_active_ids

    random_full_mean = float(random_summary["full_accuracy"]["mean"])
    random_unqueried_mean = float(random_summary["unqueried_accuracy"]["mean"])
    report = {
        "format": TRIAL_FORMAT,
        "version": TRIAL_VERSION,
        "scope": "pool-based-active-probe-selection-on-controlled-neural-teacher-domain",
        "general_world_model_claim": False,
        "autonomous_question_generation_claim": False,
        "pool_based_selection": True,
        "selector_uses_teacher_answer_before_selection": False,
        "selector_uses_expected_labels": False,
        "runtime_cognition_changed_in_v041": False,
        "domain_entities": entity_count,
        "candidate_core_facts": len(candidates),
        "query_budget": budget,
        "seed_queries": len(seed_selection),
        "active_additional_queries": len(active_extra),
        "teacher": teacher.manifest(),
        "active": {
            "teacher_queries": active_teacher.cache_misses,
            "teacher_cache_hits": active_teacher.cache_hits,
            "selection": _selection_counts(active_selected),
            "seed_selection": [c.to_dict() for c in seed_selection],
            "additional_selection": [c.to_dict() for c in active_extra],
            "frontier_diagnostics": selected_diag,
            "evaluation": active_eval,
        },
        "fixed": {
            "teacher_queries": fixed_teacher.cache_misses,
            "teacher_cache_hits": fixed_teacher.cache_hits,
            "selection": _selection_counts(fixed_selected),
            "selected": [c.to_dict() for c in fixed_selected],
            "evaluation": fixed_eval,
        },
        "random": random_summary,
        "delta_active_vs_fixed": {
            "full_accuracy": active_eval["full_accuracy"] - fixed_eval["full_accuracy"],
            "unqueried_accuracy": active_eval["unqueried_accuracy"] - fixed_eval["unqueried_accuracy"],
            "full_abstention_rate": active_eval["full_abstention_rate"] - fixed_eval["full_abstention_rate"],
        },
        "delta_active_vs_random_mean": {
            "full_accuracy": active_eval["full_accuracy"] - random_full_mean,
            "unqueried_accuracy": active_eval["unqueried_accuracy"] - random_unqueried_mean,
        },
        "controls": {
            "active_unique_teacher_queries_equal_budget": active_teacher.cache_misses == budget,
            "fixed_unique_teacher_queries_equal_budget": fixed_teacher.cache_misses == budget,
            "all_random_unique_teacher_queries_equal_budget": all(x["teacher_queries"] == budget for x in random_rows),
            "active_opaque_relation_abstention": active_eval["opaque_relation_abstention_rate"],
            "active_outside_domain_abstention": active_eval["outside_domain_abstention_rate"],
        },
        "notes": [
            "v0.41 is a selection/measurement release; RuntimeCognition remains version 0.40.",
            "The candidate question pool is known in advance. v0.41 does not generate new natural-language questions autonomously.",
            "The selector first asks one rotating relation per entity, compiles that seed BASE, then ranks the remaining pool using only Equinox abstention/uncertainty plus coverage tie-breakers.",
            "Expected labels are retained only for post-selection benchmark evaluation and are not available to the selector.",
            "A cache wraps the neural teacher: recompiling the seed prompts for the final active pack creates cache hits, not extra teacher queries.",
            "The random baseline receives the identical seed and identical total query budget; only the additional prompt selection differs.",
        ],
    }
    (out / "active-probe-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Equinox Forge v0.41 active probe selection benchmark")
    ap.add_argument("--out", default="out/v0_41_active_probe_selection")
    ap.add_argument("--entities", type=int, default=None, help="optional prefix size (min 8)")
    ap.add_argument("--budget", type=int, default=None, help="teacher query budget; default = 2 per entity")
    ap.add_argument("--random-trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=3901)
    ap.add_argument("--random-seed", type=int, default=4110)
    args = ap.parse_args(argv)
    report = run_active_probe_trial(
        args.out,
        max_entities=args.entities,
        query_budget=args.budget,
        random_trials=args.random_trials,
        seed=args.seed,
        random_seed=args.random_seed,
    )
    print(json.dumps({
        "format": report["format"],
        "version": report["version"],
        "domain_entities": report["domain_entities"],
        "query_budget": report["query_budget"],
        "active_full_accuracy": report["active"]["evaluation"]["full_accuracy"],
        "active_unqueried_accuracy": report["active"]["evaluation"]["unqueried_accuracy"],
        "fixed_full_accuracy": report["fixed"]["evaluation"]["full_accuracy"],
        "fixed_unqueried_accuracy": report["fixed"]["evaluation"]["unqueried_accuracy"],
        "random_full_accuracy_mean": report["random"]["full_accuracy"]["mean"],
        "random_unqueried_accuracy_mean": report["random"]["unqueried_accuracy"]["mean"],
        "delta_active_vs_fixed": report["delta_active_vs_fixed"],
        "delta_active_vs_random_mean": report["delta_active_vs_random_mean"],
        "controls": report["controls"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
