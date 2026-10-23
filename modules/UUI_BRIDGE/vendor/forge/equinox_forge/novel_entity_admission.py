from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import time
from typing import Any

from .baseline_gauntlet import GAUNTLET_DATA, ScriptedLadderTeacher
from .canonical import make_text_surface
from .generalization_ladder import LadderEvalCase, LadderTrainCase, compile_and_benchmark_generalization_ladder
from .ir import ForgeDocument, RelationEvidence, RelationType, SourceRef
from .runtime_cognition import RuntimeCognition, RuntimeCognitionPolicy
from .runtime_pack import load_base_pack
from .teacher_behavior import BehaviourProbeConfig

NOVEL_ENTITY_FORMAT = "equinox-forge-v0.33-novel-entity-admission-v1"
NOVEL_ENTITY_VERSION = "0.33"


@dataclass
class NovelEntityAdmissionReport:
    format: str
    version: str
    elapsed_ms: float
    headline_metrics: dict[str, Any]
    controls: dict[str, Any]
    cases: dict[str, Any]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _policy() -> RuntimeCognitionPolicy:
    return RuntimeCognitionPolicy(
        max_recall_nodes=96,
        max_recall_depth=3,
        max_predictions=3,
        min_prediction_score=0.0,
        enable_support_gated_recalled_prediction=True,
        enable_evidence_backed_novel_entity_admission=True,
        novel_entity_min_semantic_quality=0.65,
    )


def _support_document(*, include_semantics: bool) -> ForgeDocument:
    nodes = {}
    surfaces = {}
    for text in ["Wolf", "Stein", "Hund", "Katze"]:
        qs, node = make_text_surface(text)
        for q in qs:
            nodes[q.node_id] = q
        nodes[node.node_id] = node
        surfaces[text] = node

    evidence: list[RelationEvidence] = []
    if include_semantics:
        src = SourceRef("v033-independent-semantic", "1", "semantic_similarity_fixture")
        # Wolf receives strong independent semantic evidence toward the trained
        # taxonomy fillers. Stein is deliberately present in the BASE but only with
        # weak similarity, proving that lexical presence alone cannot authorize the
        # predictive substitution.
        for a, b, score in [
            ("Wolf", "Hund", 0.94),
            ("Wolf", "Katze", 0.91),
            ("Stein", "Hund", 0.08),
            ("Stein", "Katze", 0.06),
        ]:
            evidence.append(RelationEvidence(
                surfaces[a].node_id,
                surfaces[b].node_id,
                RelationType.SEMANTIC_SIMILARITY.value,
                score,
                0.98,
                src,
                {"independent_of_predictive_teacher": True},
            ))
    return ForgeDocument(nodes=sorted(nodes.values(), key=lambda n: (n.level, n.node_id)), evidence=evidence)


def _compile_track(out: Path, *, include_semantics: bool) -> tuple[dict[str, Any], RuntimeCognition]:
    train = [LadderTrainCase(*row) for row in GAUNTLET_DATA["generalization"]["train"]]
    evaluation = [LadderEvalCase(*row) for row in GAUNTLET_DATA["generalization"]["eval"]]
    report = compile_and_benchmark_generalization_ladder(
        ScriptedLadderTeacher(),
        train,
        evaluation,
        out,
        "v033-ladder",
        "1",
        probe_cfg=BehaviourProbeConfig(top_k=3, min_probability=0.0, evidence_confidence=0.95),
        cognition_policy=_policy(),
        runtime_top_k=3,
        pack_id="v033-novel-entity",
        pack_version=33,
        support_documents=[_support_document(include_semantics=include_semantics)],
    )
    cog = RuntimeCognition(load_base_pack(out / "train.eqbase"), _policy())
    return report.to_dict(), cog


def _case(cog: RuntimeCognition, prompt: str) -> dict[str, Any]:
    report = cog.predict(prompt)
    return {
        "prompt": prompt,
        "top1": report.predictions[0].surface if report.predictions else None,
        "top3": [x.surface for x in report.predictions[:3]],
        "resolved_context_ids": list(report.resolved_context_ids),
        "support_gate_checks": int(report.work.get("support_gate_checks", 0)),
        "support_gate_admissions": int(report.work.get("support_gate_admissions", 0)),
        "support_gate_rejections": int(report.work.get("support_gate_rejections", 0)),
        "semantic_admissions": int(report.work.get("support_gate_reason:evidence_backed_novel_entity_substitution", 0)),
        "semantic_rejections": int(report.work.get("support_gate_reason:insufficient_novel_entity_semantic_support", 0)),
    }


def run_novel_entity_admission_benchmark(out_dir: str | Path) -> NovelEntityAdmissionReport:
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    surface_report, surface_cog = _compile_track(out / "01_surface_only_control", include_semantics=False)
    semantic_report, semantic_cog = _compile_track(out / "02_independent_semantic_support", include_semantics=True)

    levels = {int(x["level"]): float(x["top1_retention"]) for x in semantic_report["level_reports"]}
    surface_levels = {int(x["level"]): float(x["top1_retention"]) for x in surface_report["level_reports"]}
    cases = {
        "surface_only_wolf": _case(surface_cog, "Wolf ist ein"),
        "surface_only_stone": _case(surface_cog, "Stein ist ein"),
        "semantic_wolf": _case(semantic_cog, "Wolf ist ein"),
        "semantic_stone": _case(semantic_cog, "Stein ist ein"),
        "known_recombination": _case(semantic_cog, "Hund jagt Maus"),
        "unsupported_known_mix": _case(semantic_cog, "Hund Baum"),
    }
    headline = {
        "l0_exact_replay_top1": levels.get(0, 0.0),
        "l1_suffix_transfer_top1": levels.get(1, 0.0),
        "l2_recombination_top1": levels.get(2, 0.0),
        "l3_novel_order_top1": levels.get(3, 0.0),
        "l4_evidence_backed_novel_entity_top1": levels.get(4, 0.0),
        "l5_heldout_relation_top1": levels.get(5, 0.0),
        "surface_only_l4_top1": surface_levels.get(4, 0.0),
        "wolf_semantic_admitted": cases["semantic_wolf"]["top1"] == "Tier",
        "stone_semantic_rejected": cases["semantic_stone"]["top1"] is None,
        "surface_presence_alone_rejected": cases["surface_only_wolf"]["top1"] is None,
        "known_recombination_preserved": cases["known_recombination"]["top1"] == "heute",
        "unsupported_known_mix_rejected": cases["unsupported_known_mix"]["top1"] is None,
    }
    controls = {
        "surface_only": {
            "purpose": "entity surfaces are compiled but no semantic relation authorizes substitution",
            "l4_top1": surface_levels.get(4, 0.0),
        },
        "independent_semantic_support": {
            "purpose": "non-predictive semantic evidence is compiled from a source independent of the predictive teacher",
            "l4_top1": levels.get(4, 0.0),
        },
        "predictive_train_contains_wolf": False,
        "predictive_train_contains_stein": False,
        "evaluation_teacher_predictions_compiled": False,
    }
    report = NovelEntityAdmissionReport(
        NOVEL_ENTITY_FORMAT,
        NOVEL_ENTITY_VERSION,
        (time.perf_counter() - t0) * 1000.0,
        headline,
        controls,
        cases,
        [
            "v0.33 distinguishes entity knowledge from predictive-task exposure.",
            "A surface being present somewhere in the BASE is not prediction authority.",
            "Novel predictive substitutions require compiled independent semantic evidence above threshold.",
            "The benchmark intentionally keeps L5 at zero: no new relation family is inferred by this mechanism.",
        ],
    )
    (out / "novel-entity-admission-report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Equinox Forge v0.33 evidence-backed novel-entity benchmark")
    ap.add_argument("--out", default="out/v0_33_novel_entity_admission")
    args = ap.parse_args()
    report = run_novel_entity_admission_benchmark(args.out)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
