from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .real_teacher_domain_trial import run_domain_trial
from .runtime_cognition import RuntimeCognition, RuntimeCognitionPolicy

FORMAT = "equinox-forge-v0.40-exact-context-authority-v1"
VERSION = "0.40"


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"cases": 0, "accuracy": 0.0, "abstention_rate": 0.0, "authority_activation_rate": 0.0}
    return {
        "cases": n,
        "accuracy": sum(bool(r["correct"]) for r in rows) / n,
        "abstention_rate": sum(r["top1"] is None for r in rows) / n,
        "authority_activation_rate": sum(bool(r["authority_active"]) for r in rows) / n,
        "average_recall_nodes": sum(int(r["recall_nodes"]) for r in rows) / n,
        "average_prediction_relation_checks": sum(int(r["prediction_relation_checks"]) for r in rows) / n,
        "runtime_ms_per_case": sum(float(r["runtime_ms"]) for r in rows) / n,
    }


def run_exact_context_authority_benchmark(
    out_dir: str | Path,
    *,
    max_entities: int | None = None,
    seed: int = 3901,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Reproduce the frozen v0.39 baseline and reuse the exact same compiled pack.
    baseline_dir = out / "baseline_v039"
    baseline = run_domain_trial(baseline_dir, max_entities=max_entities, seed=seed)
    pack_path = baseline_dir / "compiled" / "retention.eqbase"

    policy = RuntimeCognitionPolicy(
        max_recall_nodes=96,
        max_recall_depth=3,
        max_predictions=4,
        min_prediction_score=0.000001,
        enable_exact_context_authority=True,
        exact_context_authority_min_direct_score=0.50,
        exact_context_authority_include_suffix=True,
    )
    cognition = RuntimeCognition.from_path(pack_path, policy)

    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in baseline["cases"]:
        t0 = time.perf_counter()
        report = cognition.predict(str(case["prompt"]))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        top1 = report.predictions[0].surface if report.predictions else None
        row = {
            "category": str(case["category"]),
            "prompt": str(case["prompt"]),
            "expected": str(case["expected"]),
            "teacher_top1": case.get("teacher_top1"),
            "top1": top1,
            "correct": top1 == case["expected"],
            "matches_teacher": top1 is not None and top1 == case.get("teacher_top1"),
            "authority_active": bool(report.work.get("exact_context_authority_active", 0)),
            "authority_candidates": int(report.work.get("exact_context_authority_candidates", 0)),
            "recall_nodes": int(report.work.get("recall_nodes", 0)),
            "prediction_relation_checks": int(report.work.get("prediction_relation_checks", 0)),
            "runtime_ms": elapsed_ms,
            "top_candidates": [
                {
                    "surface": p.surface,
                    "score": p.score,
                    "direct_score": p.direct_score,
                    "recalled_score": p.recalled_score,
                    "transition_score": p.transition_score,
                }
                for p in report.predictions[:4]
            ],
        }
        rows.append(row)
        grouped[row["category"]].append(row)

    categories = {name: _summary(vals) for name, vals in sorted(grouped.items())}
    baseline_categories = {
        name: {
            "cases": int(vals["cases"]),
            "accuracy": float(vals["runtime_accuracy"]),
            "abstention_rate": float(vals["runtime_abstention_rate"]),
        }
        for name, vals in baseline["categories"].items()
    }
    deltas = {
        name: {
            "accuracy": categories[name]["accuracy"] - baseline_categories[name]["accuracy"],
            "abstention_rate": categories[name]["abstention_rate"] - baseline_categories[name]["abstention_rate"],
        }
        for name in categories
    }

    lion = next((r for r in rows if r["prompt"] == "löwe lebt im"), None)
    report = {
        "format": FORMAT,
        "version": VERSION,
        "cognition_changed": True,
        "same_compiled_pack_as_v039_baseline": True,
        "policy": {
            "enable_exact_context_authority": True,
            "exact_context_authority_min_direct_score": policy.exact_context_authority_min_direct_score,
            "exact_context_authority_include_suffix": policy.exact_context_authority_include_suffix,
        },
        "baseline_v039": baseline_categories,
        "v040": categories,
        "delta": deltas,
        "failure_trace_retest": lion,
        "controls": {
            "replay_restored_without_recompile": categories.get("replay", {}).get("accuracy") == 1.0,
            "paraphrase_restored_without_recompile": categories.get("paraphrase", {}).get("accuracy") == 1.0,
            "heldout_not_artificially_filled": categories.get("heldout_fact", {}).get("accuracy") <= 0.25,
            "opaque_relation_still_abstains": categories.get("opaque_relation", {}).get("abstention_rate") == 1.0,
            "outside_domain_still_abstains": categories.get("outside_domain_control", {}).get("abstention_rate") == 1.0,
        },
        "cases": rows,
        "baseline_report_path": str((baseline_dir / "domain-trial-report.json").resolve()),
        "pack_path": str(pack_path.resolve()),
        "notes": [
            "v0.40 reuses the exact v0.39 compiled EQBASE; no recompilation or additional teacher probing is allowed for the treatment arm.",
            "Authority activates only for exact/suffix context PREDICTION evidence above the configured direct-score floor.",
            "Weak direct evidence remains subject to ordinary aggregation; this is not an unconditional exact-wins rule.",
        ],
    }
    (out / "exact-context-authority-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Equinox Forge v0.40 exact-context authority benchmark")
    ap.add_argument("--out", default="out/v0_40_exact_context_authority")
    ap.add_argument("--entities", type=int, default=None)
    ap.add_argument("--seed", type=int, default=3901)
    args = ap.parse_args(argv)
    report = run_exact_context_authority_benchmark(args.out, max_entities=args.entities, seed=args.seed)
    print(json.dumps({
        "format": report["format"],
        "version": report["version"],
        "baseline_v039": report["baseline_v039"],
        "v040": report["v040"],
        "delta": report["delta"],
        "failure_trace_retest": report["failure_trace_retest"],
        "controls": report["controls"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
