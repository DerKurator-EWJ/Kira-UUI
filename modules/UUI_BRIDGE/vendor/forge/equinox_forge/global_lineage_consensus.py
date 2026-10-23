from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import sqlite3
import time
from typing import Any

from .canonical import make_text_surface
from .compiler import CompilePolicy, aggregate_relations
from .consensus import ConsensusPolicy, track_hypothesis_run
from .database import CanonicalDB
from .exception_consensus import ExceptionConsensusPolicy, track_exception_run
from .ir import ForgeDocument, IRNode, Modality, NodeKind, RelationEvidence, RelationType, SourceRef
from .sense_compiler import FusionWeights, SenseCompilePolicy
from .source_lineage import ensure_source_lineage_schema, register_source_lineage

GLOBAL_LINEAGE_CONSENSUS_FORMAT = "equinox-forge-v0.38-global-lineage-aware-consensus-v1"
GLOBAL_LINEAGE_CONSENSUS_VERSION = "0.38"


@dataclass
class GlobalLineageConsensusReport:
    format: str
    version: str
    elapsed_ms: float
    headline_metrics: dict[str, Any]
    relation_aggregation: dict[str, Any]
    sense_consensus: dict[str, Any]
    exception_consensus: dict[str, Any]
    controls: dict[str, bool]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _surface_nodes(*texts: str) -> tuple[list[IRNode], dict[str, IRNode]]:
    nodes: dict[str, IRNode] = {}
    surfaces: dict[str, IRNode] = {}
    for text in texts:
        qs, n = make_text_surface(text)
        for q in qs:
            nodes[q.node_id] = q
        nodes[n.node_id] = n
        surfaces[text] = n
    return sorted(nodes.values(), key=lambda n: (n.level, n.node_id)), surfaces


def _register_siblings(db: sqlite3.Connection, children: list[str], root: str) -> None:
    for child in children:
        register_source_lineage(db, child, "1", root, "1")


def _relation_aggregation_fixture(path: Path) -> dict[str, Any]:
    with CanonicalDB(path) as cdb:
        db = cdb.db
        nodes, surfaces = _surface_nodes("A", "B")
        for n in nodes:
            cdb.upsert_node(n)
        _register_siblings(db, ["sib-a", "sib-b", "sib-c"], "root-x")
        for src in ("sib-a", "sib-b", "sib-c"):
            cdb.add_evidence(RelationEvidence(
                surfaces["A"].node_id, surfaces["B"].node_id, RelationType.ASSOCIATION.value,
                0.90, 0.60, SourceRef(src, "1", "v038_relation_fixture"), {"fixture": True},
            ))
        cdb.commit()
        correlated_all = aggregate_relations(db, CompilePolicy(min_score=0, min_confidence=0, min_sources=1))
        correlated_min2 = aggregate_relations(db, CompilePolicy(min_score=0, min_confidence=0, min_sources=2))
        rel1 = next(r for r in correlated_all if r.relation_type == RelationType.ASSOCIATION.value)

        register_source_lineage(db, "ind-y", "1", "root-y", "1")
        cdb.add_evidence(RelationEvidence(
            surfaces["A"].node_id, surfaces["B"].node_id, RelationType.ASSOCIATION.value,
            0.90, 0.60, SourceRef("ind-y", "1", "v038_relation_fixture"), {"fixture": True},
        ))
        cdb.commit()
        independent_all = aggregate_relations(db, CompilePolicy(min_score=0, min_confidence=0, min_sources=1))
        independent_min2 = aggregate_relations(db, CompilePolicy(min_score=0, min_confidence=0, min_sources=2))
        rel2 = next(r for r in independent_all if r.relation_type == RelationType.ASSOCIATION.value)

        return {
            "correlated_raw_evidence_count": rel1.evidence_count,
            "correlated_effective_source_count": rel1.source_count,
            "correlated_confidence": rel1.confidence,
            "correlated_passes_min_sources_2": bool(correlated_min2),
            "independent_raw_evidence_count": rel2.evidence_count,
            "independent_effective_source_count": rel2.source_count,
            "independent_confidence": rel2.confidence,
            "independent_passes_min_sources_2": bool(independent_min2),
        }


def _make_context(target: IRNode, label: str) -> tuple[list[IRNode], IRNode]:
    qnodes, token = make_text_surface(label)
    context = IRNode(
        node_id=f"ctx_{label}", modality=Modality.TEXT.value, kind=NodeKind.COMPOSITE.value,
        level=2, surface=f"Bank {label}", members=[target.node_id, token.node_id], metadata={"fixture": True},
    )
    return qnodes + [token], context


def _sense_consensus_fixture(path: Path) -> dict[str, Any]:
    with CanonicalDB(path) as cdb:
        db = cdb.db
        base_nodes, surfaces = _surface_nodes("Bank")
        target = surfaces["Bank"]
        for n in base_nodes:
            cdb.upsert_node(n)
        contexts: dict[str, IRNode] = {}
        for label in ("finance_a", "finance_b", "seat_a", "seat_b"):
            ns, ctx = _make_context(target, label)
            for n in ns:
                cdb.upsert_node(n)
            cdb.upsert_node(ctx)
            contexts[label] = ctx
            cdb.add_evidence(RelationEvidence(
                target.node_id, ctx.node_id, RelationType.CONTEXTUAL_USE.value,
                1.0, 1.0, SourceRef("structure", "1", "v038_structure"), {"fixture": True},
            ))

        finance = [contexts["finance_a"].node_id, contexts["finance_b"].node_id]
        seat = [contexts["seat_a"].node_id, contexts["seat_b"].node_id]
        sense_nodes = [
            IRNode("sense_finance", Modality.TEXT.value, NodeKind.CONCEPT.value, 3, surface="finance-sense",
                   members=[target.node_id, *finance], metadata={"provisional": True, "target_id": target.node_id, "contexts": finance}),
            IRNode("sense_seat", Modality.TEXT.value, NodeKind.CONCEPT.value, 3, surface="seat-sense",
                   members=[target.node_id, *seat], metadata={"provisional": True, "target_id": target.node_id, "contexts": seat}),
        ]
        doc = ForgeDocument(nodes=sense_nodes, evidence=[])
        _register_siblings(db, ["sib-a", "sib-b", "sib-c"], "root-x")

        def add_support(src: str) -> None:
            ref = SourceRef(src, "1", "v038_sense_fixture")
            for group in (finance, seat):
                cdb.add_evidence(RelationEvidence(group[0], group[1], RelationType.CONTEXTUAL_SIMILARITY.value, 0.95, 1.0, ref, {"fixture": True}))

        for src in ("sib-a", "sib-b", "sib-c"):
            add_support(src)
        cdb.commit()
        sp = SenseCompilePolicy(
            threshold=0.75, min_channels=1, min_contexts=2,
            weights=FusionWeights(hidden_similarity=1.0, lexical_overlap=0.0, semantic_similarity=0.0,
                                  substitution=0.0, transition=0.0, prediction_signature=0.0),
        )
        cp = ConsensusPolicy(
            source_vote_threshold=0.58, source_vote_margin=0.08, min_observed_pairs=1,
            stable_min_sources=2, stable_min_runs=2, stable_consensus=0.75,
            stable_min_support_weight=1.5, quarantine_conflicts=2, quarantine_conflict_weight=1.5,
            quarantine_after_misses=2,
        )
        r1 = track_hypothesis_run(db, doc, target.node_id, sp, cp)
        r2 = track_hypothesis_run(db, doc, target.node_id, sp, cp)
        correlated_statuses = sorted({h["status"] for h in r2.hypotheses})
        correlated_support = sorted({h["support_sources"] for h in r2.hypotheses})

        register_source_lineage(db, "ind-y", "1", "root-y", "1")
        add_support("ind-y")
        cdb.commit()
        r3 = track_hypothesis_run(db, doc, target.node_id, sp, cp)
        r4 = track_hypothesis_run(db, doc, target.node_id, sp, cp)
        independent_statuses = sorted({h["status"] for h in r4.hypotheses})
        independent_support = sorted({h["support_sources"] for h in r4.hypotheses})
        return {
            "correlated_after_two_runs": correlated_statuses,
            "correlated_effective_support_counts": correlated_support,
            "independent_after_two_more_runs": independent_statuses,
            "independent_effective_support_counts": independent_support,
            "final_votes": r4.hypotheses[0]["votes"] if r4.hypotheses else [],
        }


def _exception_consensus_fixture(path: Path) -> dict[str, Any]:
    with CanonicalDB(path) as cdb:
        db = cdb.db
        nodes, surfaces = _surface_nodes("ctx-a", "ctx-b", "Sauerstoff", "Luft")
        for n in nodes:
            cdb.upsert_node(n)
        override = IRNode(
            "override_fixture", Modality.TEXT.value, NodeKind.TEMPLATE.value, 3, surface="override",
            metadata={
                "exception_override": True,
                "support_context_ids": [surfaces["ctx-a"].node_id, surfaces["ctx-b"].node_id],
                "exception_top_target_id": surfaces["Sauerstoff"].node_id,
                "exception_general_top_target_id": surfaces["Luft"].node_id,
                "overridden_frame_node_id": "general-frame",
                "exception_subtype_prototype_node_id": "bird-prototype",
            },
        )
        cdb.upsert_node(override)
        doc = ForgeDocument(nodes=[override], evidence=[])
        _register_siblings(db, ["sib-a", "sib-b", "sib-c"], "root-x")

        def add_support(src: str) -> None:
            ref = SourceRef(src, "1", "v038_exception_fixture")
            for ctx in (surfaces["ctx-a"], surfaces["ctx-b"]):
                cdb.add_evidence(RelationEvidence(ctx.node_id, surfaces["Sauerstoff"].node_id, RelationType.PREDICTION.value, 0.90, 1.0, ref, {"fixture": True}))
                cdb.add_evidence(RelationEvidence(ctx.node_id, surfaces["Luft"].node_id, RelationType.PREDICTION.value, 0.10, 1.0, ref, {"fixture": True}))

        for src in ("sib-a", "sib-b", "sib-c"):
            add_support(src)
        cdb.commit()
        policy = ExceptionConsensusPolicy(
            source_min_context_fraction=1.0, source_vote_threshold=0.55, source_vote_margin=0.15,
            stable_min_sources=2, stable_min_runs=2, stable_consensus=0.75,
            stable_min_support_weight=1.5, quarantine_conflicts=2, quarantine_conflict_weight=1.5,
            quarantine_after_misses=2,
        )
        r1 = track_exception_run(db, doc, policy)
        r2 = track_exception_run(db, doc, policy)
        correlated = r2.assessments[0]

        register_source_lineage(db, "ind-y", "1", "root-y", "1")
        add_support("ind-y")
        cdb.commit()
        r3 = track_exception_run(db, doc, policy)
        r4 = track_exception_run(db, doc, policy)
        independent = r4.assessments[0]
        return {
            "correlated_after_two_runs": correlated["status"],
            "correlated_effective_support_count": correlated["support_sources"],
            "independent_after_two_more_runs": independent["status"],
            "independent_effective_support_count": independent["support_sources"],
            "final_votes": independent["votes"],
        }


def run_global_lineage_consensus_benchmark(out_dir: str | Path) -> GlobalLineageConsensusReport:
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    relation = _relation_aggregation_fixture(out / "relation.sqlite")
    sense = _sense_consensus_fixture(out / "sense.sqlite")
    exception = _exception_consensus_fixture(out / "exception.sqlite")

    controls = {
        "relation_siblings_count_once": relation["correlated_effective_source_count"] == 1,
        "relation_correlated_confidence_not_noisy_or_tripled": abs(relation["correlated_confidence"] - 0.60) < 1e-9,
        "relation_correlated_fails_min_sources_2": relation["correlated_passes_min_sources_2"] is False,
        "relation_independent_root_restores_min_sources_2": relation["independent_effective_source_count"] == 2 and relation["independent_passes_min_sources_2"] is True,
        "sense_correlated_support_cannot_stabilize": sense["correlated_after_two_runs"] == ["candidate"] and sense["correlated_effective_support_counts"] == [1],
        "sense_independent_lineage_can_stabilize": sense["independent_after_two_more_runs"] == ["stable"] and sense["independent_effective_support_counts"] == [2],
        "exception_correlated_support_cannot_stabilize": exception["correlated_after_two_runs"] == "candidate" and exception["correlated_effective_support_count"] == 1,
        "exception_independent_lineage_can_stabilize": exception["independent_after_two_more_runs"] == "stable" and exception["independent_effective_support_count"] == 2,
    }
    headline = {
        "relation_correlated_effective_sources": relation["correlated_effective_source_count"],
        "relation_correlated_confidence": relation["correlated_confidence"],
        "relation_independent_effective_sources": relation["independent_effective_source_count"],
        "relation_independent_confidence": relation["independent_confidence"],
        "sense_correlated_status": sense["correlated_after_two_runs"],
        "sense_independent_status": sense["independent_after_two_more_runs"],
        "exception_correlated_status": exception["correlated_after_two_runs"],
        "exception_independent_status": exception["independent_after_two_more_runs"],
        "all_controls_pass": all(controls.values()),
    }
    report = GlobalLineageConsensusReport(
        format=GLOBAL_LINEAGE_CONSENSUS_FORMAT,
        version=GLOBAL_LINEAGE_CONSENSUS_VERSION,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        headline_metrics=headline,
        relation_aggregation=relation,
        sense_consensus=sense,
        exception_consensus=exception,
        controls=controls,
        notes=[
            "v0.38 extends source-lineage semantics from composition induction/lifecycle into general compiled-relation aggregation, sense consensus, and exception consensus.",
            "Correlated descendants count as one effective evidence line; source aliases cannot satisfy min_sources by renaming the same ancestry.",
            "Compiled relation confidence is combined across independent lineage components, not raw correlated evidence rows.",
            "Within a correlated lineage, support/conflict disagreement becomes abstention rather than an intra-family majority vote.",
            "Databases without lineage metadata retain the pre-v0.38 aggregation and consensus behavior for backward compatibility.",
        ],
    )
    (out / "global-lineage-consensus-report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Equinox Forge v0.38 global lineage-aware consensus benchmark")
    ap.add_argument("--out", default="out/v0_38_global_lineage_consensus")
    args = ap.parse_args()
    print(json.dumps(run_global_lineage_consensus_benchmark(args.out).to_dict(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
