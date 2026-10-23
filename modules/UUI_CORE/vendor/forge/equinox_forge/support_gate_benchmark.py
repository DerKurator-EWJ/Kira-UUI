from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import time
from typing import Any

from .baseline_gauntlet import gauntlet_dataset_sha256, run_generalization_and_rejection, run_typed_frame_track
from .runtime_cognition import RuntimeCognition, RuntimeCognitionPolicy
from .runtime_pack import load_base_pack

SUPPORT_GATE_FORMAT = "equinox-forge-v0.32-support-gate-benchmark-v1"
SUPPORT_GATE_VERSION = "0.32"

V031_REFERENCE = {
    "exact_replay_top1": 1.0,
    "known_suffix_transfer_top1": 1.0,
    "known_parts_recombination_top1": 1.0,
    "novel_order_known_lexicon_top1": 1.0,
    "novel_entity_known_frame_top1": 1.0,
    "heldout_relation_family_top1": 0.0,
    "unknown_rejection_rate": 0.25,
    "known_or_mixed_unsupported_rejection_rate": 0.0,
}


@dataclass
class SupportGateBenchmarkReport:
    format: str
    version: str
    dataset_sha256: str
    elapsed_ms: float
    headline_metrics: dict[str, Any]
    delta_vs_v031: dict[str, float]
    tradeoffs: list[str]
    indistinguishability_case: dict[str, Any]
    generalization_and_rejection: dict[str, Any]
    typed_frame_admission: dict[str, Any]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _policy(*, support_gate: bool = True) -> RuntimeCognitionPolicy:
    return RuntimeCognitionPolicy(
        max_recall_nodes=96,
        max_recall_depth=3,
        max_predictions=3,
        min_prediction_score=0.0,
        enable_support_gated_recalled_prediction=support_gate,
    )


def _surface_signature(cog: RuntimeCognition, prompt: str) -> dict[str, Any]:
    report = cog.predict(prompt)
    live_tokens = prompt.split()
    known = []
    unknown = []
    for tok in live_tokens:
        ids = [
            nid for nid in cog.index.surface_to_nodes.get(tok, [])
            if cog.index.node_by_id[nid].kind == "surface"
        ]
        (known if ids else unknown).append(tok)
    recalled_predictive = []
    for trace in report.recall:
        node = cog.index.node_by_id.get(trace.node_id)
        if node is None or not node.surface:
            continue
        if not any(r.relation_type == "prediction" for r in cog.index.relations_from.get(trace.node_id, [])):
            continue
        shared = sorted(set(live_tokens) & set(node.surface.split()))
        if shared:
            recalled_predictive.append({
                "surface": node.surface,
                "shared_tokens": shared,
                "shared_count": len(shared),
                "activation": trace.activation,
            })
    recalled_predictive.sort(key=lambda x: (-x["shared_count"], -x["activation"], x["surface"]))
    return {
        "prompt": prompt,
        "known_tokens": known,
        "unknown_tokens": unknown,
        "resolved_context_ids": list(report.resolved_context_ids),
        "top1": report.predictions[0].surface if report.predictions else None,
        "recalled_predictive_contexts": recalled_predictive,
    }


def run_support_gate_benchmark(out_dir: str | Path) -> SupportGateBenchmarkReport:
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    general = run_generalization_and_rejection(out / "01_generalization_and_rejection", _policy(support_gate=True))
    typed = run_typed_frame_track(out / "02_typed_frame_admission")

    level_top1 = {int(x["level"]): float(x["top1_retention"]) for x in general["ladder"]["level_reports"]}
    rejection = general["unknown_rejection"]
    headline = {
        "exact_replay_top1": level_top1.get(0, 0.0),
        "known_suffix_transfer_top1": level_top1.get(1, 0.0),
        "known_parts_recombination_top1": level_top1.get(2, 0.0),
        "novel_order_known_lexicon_top1": level_top1.get(3, 0.0),
        "novel_entity_known_frame_top1": level_top1.get(4, 0.0),
        "heldout_relation_family_top1": level_top1.get(5, 0.0),
        "unknown_rejection_rate": rejection["unknown_rejection_rate"],
        "known_or_mixed_unsupported_rejection_rate": rejection["known_or_mixed_unsupported_rejection_rate"],
        "typed_frame_precision": typed["typed_frame_precision"],
        "typed_frame_recall": typed["typed_frame_recall"],
    }
    delta = {
        key: float(headline[key]) - float(value)
        for key, value in V031_REFERENCE.items()
    }

    pack = load_base_pack(out / "01_generalization_and_rejection" / "ladder" / "train.eqbase")
    gated = RuntimeCognition(pack, _policy(support_gate=True))
    legacy = RuntimeCognition(pack, _policy(support_gate=False))
    wolf_gated = _surface_signature(gated, "Wolf ist ein")
    stone_gated = _surface_signature(gated, "Stein ist ein")
    wolf_legacy = _surface_signature(legacy, "Wolf ist ein")
    stone_legacy = _surface_signature(legacy, "Stein ist ein")
    indistinguishability = {
        "claim": "Without external semantic admission, Wolf ist ein and Stein ist ein expose the same runtime support shape: one unseen entity plus the known relational suffix ist ein.",
        "gated": {"wolf": wolf_gated, "stone": stone_gated},
        "legacy": {"wolf": wolf_legacy, "stone": stone_legacy},
        "consequence": "v0.32 abstains on both. Novel-entity transfer requires an explicit admitted frame/prototype/hierarchy path instead of lexical recall alone.",
    }

    tradeoffs = []
    if headline["novel_entity_known_frame_top1"] < V031_REFERENCE["novel_entity_known_frame_top1"]:
        tradeoffs.append(
            "L4 lexical-only novel-entity retention drops from 1.0 to 0.0. This is intentional: the previous success is not distinguishable from the Stein false positive without semantic admission."
        )
    if headline["heldout_relation_family_top1"] < 1.0:
        tradeoffs.append("Held-out relation-family transfer remains unsupported; support gating does not invent missing knowledge.")

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    report = SupportGateBenchmarkReport(
        format=SUPPORT_GATE_FORMAT,
        version=SUPPORT_GATE_VERSION,
        dataset_sha256=gauntlet_dataset_sha256(),
        elapsed_ms=elapsed_ms,
        headline_metrics=headline,
        delta_vs_v031=delta,
        tradeoffs=tradeoffs,
        indistinguishability_case=indistinguishability,
        generalization_and_rejection=general,
        typed_frame_admission=typed,
        notes=[
            "Recall remains broad; only the authority to execute recalled PREDICTION relations is gated.",
            "Direct exact/suffix contexts, transitions and admitted frame-schema predictions retain their separate execution paths.",
            "The v0.31 case set and dataset hash are reused unchanged so deltas are attributable to runtime semantics rather than benchmark drift.",
            "No composite score is emitted.",
        ],
    )
    (out / "support-gate-report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run Equinox Forge v0.32 Support-Gated Prediction benchmark")
    p.add_argument("--out", default="out/v0_32_support_gate", help="output directory")
    p.add_argument("--json", action="store_true", help="print full JSON report")
    args = p.parse_args(argv)
    report = run_support_gate_benchmark(args.out)
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(json.dumps({
            "format": report.format,
            "version": report.version,
            "dataset_sha256": report.dataset_sha256,
            "headline_metrics": report.headline_metrics,
            "delta_vs_v031": report.delta_vs_v031,
            "tradeoffs": report.tradeoffs,
            "report": str(Path(args.out) / "support-gate-report.json"),
        }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
