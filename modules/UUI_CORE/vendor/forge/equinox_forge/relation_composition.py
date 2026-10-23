from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import time
from typing import Any

from .baseline_gauntlet import GAUNTLET_DATA, ScriptedLadderTeacher
from .canonical import make_composite, make_text_surface
from .generalization_ladder import LadderEvalCase, LadderTrainCase, compile_and_benchmark_generalization_ladder
from .ir import ForgeDocument, IRNode, NodeKind, RelationEvidence, RelationType, SourceRef
from .novel_entity_admission import _support_document as novel_entity_support_document
from .runtime_cognition import RuntimeCognition, RuntimeCognitionPolicy
from .runtime_pack import load_base_pack
from .teacher_behavior import BehaviourProbeConfig

RELATION_COMPOSITION_FORMAT = "equinox-forge-v0.34-relation-composition-v1"
RELATION_COMPOSITION_VERSION = "0.34"


@dataclass
class RelationCompositionReport:
    format: str
    version: str
    elapsed_ms: float
    headline_metrics: dict[str, Any]
    controls: dict[str, Any]
    cases: dict[str, Any]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _policy(*, composition: bool = True) -> RuntimeCognitionPolicy:
    return RuntimeCognitionPolicy(
        max_recall_nodes=96,
        max_recall_depth=3,
        max_predictions=3,
        min_prediction_score=0.0,
        enable_support_gated_recalled_prediction=True,
        enable_evidence_backed_novel_entity_admission=True,
        novel_entity_min_semantic_quality=0.65,
        enable_relation_composition_prediction=composition,
        relation_composition_weight=0.86,
        relation_composition_min_path_quality=0.50,
    )


def _merge_documents(*docs: ForgeDocument) -> ForgeDocument:
    nodes: dict[str, IRNode] = {}
    evidence: list[RelationEvidence] = []
    for doc in docs:
        for node in doc.nodes:
            existing = nodes.get(node.node_id)
            if existing is not None and (
                existing.modality != node.modality
                or existing.kind != node.kind
                or existing.level != node.level
                or existing.surface != node.surface
                or existing.members != node.members
            ):
                raise ValueError(f"support document node collision: {node.node_id}")
            if existing is None:
                nodes[node.node_id] = node
            elif node.metadata and not existing.metadata:
                nodes[node.node_id] = node
        evidence.extend(doc.evidence)
    return ForgeDocument(nodes=sorted(nodes.values(), key=lambda n: (n.level, n.node_id)), evidence=evidence)


def _composition_support_document(*, include_schema: bool, include_path: bool) -> ForgeDocument:
    """Build non-predictive support for the frozen L5 prompt `Hund wohnt in`.

    No PREDICTION or TRANSITION edge for the held-out family is added. The positive
    track contains only a stable composition schema and a semantic two-hop path:

        Hund --is_a--> Haustier --property--> Haus

    The schema maps `[SLOT_0] wohnt in` to that path. Removing either the schema or
    the path must restore abstention.
    """
    nodes: dict[str, IRNode] = {}
    surfaces: dict[str, IRNode] = {}
    for text in ["Hund", "Stein", "Haustier", "Objekt", "Haus", "[SLOT_0]", "wohnt", "in"]:
        qs, node = make_text_surface(text)
        for q in qs:
            nodes[q.node_id] = q
        nodes[node.node_id] = node
        surfaces[text] = node

    evidence: list[RelationEvidence] = []
    if include_path:
        src = SourceRef("v034-semantic-path", "1", "relation_composition_fixture")
        evidence.extend([
            RelationEvidence(
                surfaces["Hund"].node_id,
                surfaces["Haustier"].node_id,
                RelationType.IS_A.value,
                0.96,
                0.97,
                src,
                {"role": "composition_premise", "step": 1, "predictive_family": False},
            ),
            RelationEvidence(
                surfaces["Haustier"].node_id,
                surfaces["Haus"].node_id,
                RelationType.PROPERTY.value,
                0.92,
                0.96,
                src,
                {"role": "composition_premise", "step": 2, "predictive_family": False},
            ),
            # Negative control entity is semantically typed but lacks the second
            # premise, proving that pattern match alone does not emit Haus.
            RelationEvidence(
                surfaces["Stein"].node_id,
                surfaces["Objekt"].node_id,
                RelationType.IS_A.value,
                0.97,
                0.98,
                src,
                {"role": "composition_negative_control", "predictive_family": False},
            ),
        ])

    if include_schema:
        schema = make_composite(
            "text",
            [surfaces["[SLOT_0]"], surfaces["wohnt"], surfaces["in"]],
            kind=NodeKind.TEMPLATE.value,
            label="[SLOT_0] wohnt in",
        )
        schema.metadata = {
            "role": "relation_composition_schema",
            "pattern_tokens": ["[SLOT_0]", "wohnt", "in"],
            "path": [
                {"relation_type": RelationType.IS_A.value, "direction": "out"},
                {"relation_type": RelationType.PROPERTY.value, "direction": "out"},
            ],
            "status": "stable",
            "support_count": 1,
            "source_count": 1,
            "min_path_quality": 0.50,
            "direct_prediction_family": False,
            "fixture": "v0.34",
        }
        nodes[schema.node_id] = schema

    return ForgeDocument(nodes=sorted(nodes.values(), key=lambda n: (n.level, n.node_id)), evidence=evidence)


def _compile_track(out: Path, *, include_schema: bool, include_path: bool, composition_enabled: bool = True) -> tuple[dict[str, Any], RuntimeCognition]:
    train = [LadderTrainCase(*row) for row in GAUNTLET_DATA["generalization"]["train"]]
    evaluation = [LadderEvalCase(*row) for row in GAUNTLET_DATA["generalization"]["eval"]]
    support = _merge_documents(
        novel_entity_support_document(include_semantics=True),
        _composition_support_document(include_schema=include_schema, include_path=include_path),
    )
    policy = _policy(composition=composition_enabled)
    report = compile_and_benchmark_generalization_ladder(
        ScriptedLadderTeacher(),
        train,
        evaluation,
        out,
        "v034-ladder",
        "1",
        probe_cfg=BehaviourProbeConfig(top_k=3, min_probability=0.0, evidence_confidence=0.95),
        cognition_policy=policy,
        runtime_top_k=3,
        pack_id="v034-relation-composition",
        pack_version=34,
        support_documents=[support],
        include_pack_metadata=True,
    )
    return report.to_dict(), RuntimeCognition(load_base_pack(out / "train.eqbase"), policy)


def _case(cog: RuntimeCognition, prompt: str) -> dict[str, Any]:
    report = cog.predict(prompt)
    return {
        "prompt": prompt,
        "top1": report.predictions[0].surface if report.predictions else None,
        "top3": [x.surface for x in report.predictions[:3]],
        "resolved_context_ids": list(report.resolved_context_ids),
        "composition_schema_checks": int(report.work.get("relation_composition_schema_checks", 0)),
        "composition_matches": int(report.work.get("relation_composition_matches", 0)),
        "composition_admissions": int(report.work.get("relation_composition_admissions", 0)),
        "composition_rejections": int(report.work.get("relation_composition_rejections", 0)),
        "composition_path_relation_checks": int(report.work.get("relation_composition_path_relation_checks", 0)),
        "relation_composition": list(report.relation_composition),
    }


def _levels(report: dict[str, Any]) -> dict[int, float]:
    return {int(x["level"]): float(x["top1_retention"]) for x in report["level_reports"]}


def run_relation_composition_benchmark(out_dir: str | Path) -> RelationCompositionReport:
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    opaque_report, opaque_cog = _compile_track(out / "01_opaque_control", include_schema=False, include_path=False)
    graph_report, graph_cog = _compile_track(out / "02_graph_only_control", include_schema=False, include_path=True)
    schema_report, schema_cog = _compile_track(out / "03_schema_only_control", include_schema=True, include_path=False)
    full_report, full_cog = _compile_track(out / "04_schema_plus_path", include_schema=True, include_path=True)
    ablated_report, ablated_cog = _compile_track(
        out / "05_runtime_ablation", include_schema=True, include_path=True, composition_enabled=False
    )

    full_levels = _levels(full_report)
    opaque_levels = _levels(opaque_report)
    graph_levels = _levels(graph_report)
    schema_levels = _levels(schema_report)
    ablated_levels = _levels(ablated_report)

    cases = {
        "opaque_hund": _case(opaque_cog, "Hund wohnt in"),
        "graph_only_hund": _case(graph_cog, "Hund wohnt in"),
        "schema_only_hund": _case(schema_cog, "Hund wohnt in"),
        "composed_hund": _case(full_cog, "Hund wohnt in"),
        "composed_stein": _case(full_cog, "Stein wohnt in"),
        "unmatched_relation_phrase": _case(full_cog, "Hund wohnt aus"),
        "runtime_ablation_hund": _case(ablated_cog, "Hund wohnt in"),
    }

    headline = {
        "l0_exact_replay_top1": full_levels.get(0, 0.0),
        "l1_suffix_transfer_top1": full_levels.get(1, 0.0),
        "l2_recombination_top1": full_levels.get(2, 0.0),
        "l3_novel_order_top1": full_levels.get(3, 0.0),
        "l4_evidence_backed_novel_entity_top1": full_levels.get(4, 0.0),
        "l5_composed_heldout_prediction_family_top1": full_levels.get(5, 0.0),
        "opaque_l5_top1": opaque_levels.get(5, 0.0),
        "graph_only_l5_top1": graph_levels.get(5, 0.0),
        "schema_only_l5_top1": schema_levels.get(5, 0.0),
        "runtime_ablation_l5_top1": ablated_levels.get(5, 0.0),
        "composed_hund_is_haus": cases["composed_hund"]["top1"] == "Haus",
        "stein_without_complete_path_abstains": cases["composed_stein"]["top1"] is None,
        "unmatched_relation_phrase_abstains": cases["unmatched_relation_phrase"]["top1"] is None,
    }
    controls = {
        "predictive_train_contains_hund_wohnt_in": False,
        "evaluation_teacher_predictions_compiled": False,
        "direct_prediction_edge_for_hund_wohnt_in": False,
        "opaque_relation_family_remains_unsupported": opaque_levels.get(5, 0.0) == 0.0,
        "semantic_graph_without_schema_is_not_prediction_authority": graph_levels.get(5, 0.0) == 0.0,
        "schema_without_complete_path_is_not_prediction_authority": schema_levels.get(5, 0.0) == 0.0,
        "runtime_switch_ablation_restores_abstention": ablated_levels.get(5, 0.0) == 0.0,
    }
    report = RelationCompositionReport(
        RELATION_COMPOSITION_FORMAT,
        RELATION_COMPOSITION_VERSION,
        (time.perf_counter() - t0) * 1000.0,
        headline,
        controls,
        cases,
        [
            "v0.34 does not infer an opaque relation family from lexical similarity.",
            "The frozen L5 prompt is recovered only when an explicit stable composition schema and every non-predictive semantic premise are compiled into the BASE.",
            "The held-out property is direct predictive exposure: the L5 prompt and its teacher prediction are not compiled.",
            "Graph-only, schema-only, runtime-ablation, incomplete-path, and unmatched-pattern controls must abstain.",
            "The schema is fixture-authored in v0.34. Automatic induction of composition schemas is deliberately a later research problem.",
        ],
    )
    (out / "relation-composition-report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Equinox Forge v0.34 evidence-backed relation-composition benchmark")
    ap.add_argument("--out", default="out/v0_34_relation_composition")
    args = ap.parse_args()
    report = run_relation_composition_benchmark(args.out)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
