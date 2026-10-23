from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import shutil
import time
from typing import Any, Iterable

from .baseline_gauntlet import GAUNTLET_DATA, ScriptedLadderTeacher
from .canonical import make_composite, make_text_surface
from .generalization_ladder import LadderEvalCase, LadderTrainCase, compile_and_benchmark_generalization_ladder
from .ir import ForgeDocument, IRNode, NodeKind, RelationEvidence, RelationType, SourceRef
from .novel_entity_admission import _support_document as novel_entity_support_document
from .runtime_cognition import RuntimeCognition, RuntimeCognitionPolicy
from .runtime_pack import load_base_pack
from .source_lineage import LineageComponent, SourceLineageGraph
from .teacher_behavior import BehaviourProbeConfig

RELATION_COMPOSITION_INDUCTION_FORMAT = "equinox-forge-v0.35-relation-composition-induction-v1"
RELATION_COMPOSITION_INDUCTION_VERSION = "0.35"


@dataclass(frozen=True)
class RelationCompositionObservation:
    prompt: str
    target_surface: str
    source: SourceRef


@dataclass(frozen=True)
class RelationCompositionInductionPolicy:
    allowed_premise_types: tuple[str, ...] = (
        RelationType.IS_A.value,
        RelationType.PROPERTY.value,
        RelationType.ACTION.value,
    )
    max_hops: int = 3
    min_examples: int = 3
    min_distinct_fillers: int = 3
    min_support: int = 3
    min_sources: int = 2
    min_coverage: float = 0.75
    min_mean_path_quality: float = 0.50
    min_support_margin: int = 1
    runtime_min_path_quality: float = 0.50


@dataclass
class InducedCompositionCandidate:
    pattern_tokens: list[str]
    slot_position: int
    path: list[dict[str, str]]
    support_count: int
    example_count: int
    source_count: int
    raw_source_count: int
    distinct_filler_count: int
    coverage: float
    mean_path_quality: float
    min_path_quality_observed: float
    support_margin: int
    status: str
    supporting_prompts: list[str] = field(default_factory=list)
    supporting_targets: list[str] = field(default_factory=list)
    supporting_sources: list[str] = field(default_factory=list)
    supporting_lineage_components: list[list[str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RelationCompositionInductionResult:
    schemas: ForgeDocument
    candidates: list[InducedCompositionCandidate]
    group_count: int
    observation_count: int
    stable_count: int
    contested_count: int
    candidate_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_count": self.group_count,
            "observation_count": self.observation_count,
            "stable_count": self.stable_count,
            "contested_count": self.contested_count,
            "candidate_count": self.candidate_count,
            "candidates": [x.to_dict() for x in self.candidates],
            "schemas": self.schemas.to_dict(),
        }


@dataclass
class RelationCompositionInductionBenchmarkReport:
    format: str
    version: str
    elapsed_ms: float
    headline_metrics: dict[str, Any]
    induction: dict[str, Any]
    controls: dict[str, Any]
    cases: dict[str, Any]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _surface_nodes(document: ForgeDocument) -> dict[str, list[IRNode]]:
    out: dict[str, list[IRNode]] = {}
    for node in document.nodes:
        if node.kind == NodeKind.QUANT.value or not node.surface:
            continue
        out.setdefault(node.surface, []).append(node)
    return out


def _edge_index(document: ForgeDocument, allowed: set[str]) -> dict[str, list[tuple[str, str, float]]]:
    """Return best outgoing evidence edges as (target, relation_type, quality)."""
    best: dict[tuple[str, str, str], float] = {}
    for ev in document.evidence:
        if ev.relation_type not in allowed:
            continue
        q = max(0.0, min(1.0, float(ev.score) * float(ev.confidence)))
        key = (ev.from_id, ev.to_id, ev.relation_type)
        if q > best.get(key, -1.0):
            best[key] = q
    out: dict[str, list[tuple[str, str, float]]] = {}
    for (src, dst, rtype), q in best.items():
        out.setdefault(src, []).append((dst, rtype, q))
    for src in out:
        out[src].sort(key=lambda x: (x[1], x[0], -x[2]))
    return out


def _path_signatures(
    starts: Iterable[str],
    targets: set[str],
    edges: dict[str, list[tuple[str, str, float]]],
    max_hops: int,
) -> dict[tuple[tuple[str, str], ...], float]:
    """Find simple outgoing relation-type paths and keep best quality per signature."""
    found: dict[tuple[tuple[str, str], ...], float] = {}
    frontier: list[tuple[str, float, tuple[str, ...], tuple[tuple[str, str], ...]]] = [
        (sid, 1.0, (sid,), ()) for sid in starts
    ]
    for _depth in range(1, max_hops + 1):
        nxt_frontier: list[tuple[str, float, tuple[str, ...], tuple[tuple[str, str], ...]]] = []
        for current, quality, visited, signature in frontier:
            for nxt, rtype, edge_q in edges.get(current, []):
                if nxt in visited:
                    continue
                nq = quality * edge_q
                nsig = signature + ((rtype, "out"),)
                if nxt in targets:
                    found[nsig] = max(found.get(nsig, 0.0), nq)
                nxt_frontier.append((nxt, nq, visited + (nxt,), nsig))
        frontier = nxt_frontier
        if not frontier:
            break
    return found


def _schema_node(candidate: InducedCompositionCandidate, policy: RelationCompositionInductionPolicy) -> IRNode:
    members: list[IRNode] = []
    for token in candidate.pattern_tokens:
        _qs, surface = make_text_surface(token)
        members.append(surface)
    path_label = ">".join(step["relation_type"] for step in candidate.path)
    label = f"{' '.join(candidate.pattern_tokens)} :: {path_label}"
    schema = make_composite("text", members, kind=NodeKind.TEMPLATE.value, label=label)
    schema.metadata = {
        "role": "relation_composition_schema",
        "pattern_tokens": list(candidate.pattern_tokens),
        "path": list(candidate.path),
        "status": candidate.status,
        "support_count": candidate.support_count,
        "source_count": candidate.source_count,
        "raw_source_count": candidate.raw_source_count,
        "supporting_lineage_components": [list(x) for x in candidate.supporting_lineage_components],
        "example_count": candidate.example_count,
        "distinct_filler_count": candidate.distinct_filler_count,
        "coverage": candidate.coverage,
        "mean_path_quality": candidate.mean_path_quality,
        "support_margin": candidate.support_margin,
        "min_path_quality": policy.runtime_min_path_quality,
        "direct_prediction_family": False,
        "induced": True,
        "induction_version": RELATION_COMPOSITION_INDUCTION_VERSION,
        "supporting_prompts": list(candidate.supporting_prompts),
        "supporting_targets": list(candidate.supporting_targets),
        "supporting_sources": list(candidate.supporting_sources),
    }
    return schema


def induce_relation_composition_schemas(
    semantic_document: ForgeDocument,
    observations: list[RelationCompositionObservation],
    policy: RelationCompositionInductionPolicy | None = None,
    lineage_graph: SourceLineageGraph | None = None,
) -> RelationCompositionInductionResult:
    """Induce one-slot relation-composition programs from repeated direct observations.

    The relation phrase is not supplied. Candidate prompt patterns are discovered by
    replacing each token position in turn with a slot and grouping observations that
    share the remaining literal tokens. Semantic graph paths then provide candidate
    programs from the varying slot filler to the observed target.
    """
    policy = policy or RelationCompositionInductionPolicy()
    lineage_graph = lineage_graph or SourceLineageGraph()
    surface = _surface_nodes(semantic_document)
    edges = _edge_index(semantic_document, set(policy.allowed_premise_types))

    groups: dict[tuple[str, ...], list[RelationCompositionObservation]] = {}
    slot_positions: dict[tuple[str, ...], int] = {}
    for obs in observations:
        tokens = [x for x in obs.prompt.split() if x]
        for pos in range(len(tokens)):
            pattern = list(tokens)
            pattern[pos] = "[SLOT_0]"
            key = tuple(pattern)
            groups.setdefault(key, []).append(obs)
            slot_positions[key] = pos

    candidate_rows: list[InducedCompositionCandidate] = []
    eligible_group_count = 0
    for pattern, rows in sorted(groups.items()):
        pos = slot_positions[pattern]
        fillers = [r.prompt.split()[pos] for r in rows if len(r.prompt.split()) == len(pattern)]
        distinct_fillers = sorted(set(fillers))
        if len(rows) < policy.min_examples or len(distinct_fillers) < policy.min_distinct_fillers:
            continue
        eligible_group_count += 1

        per_signature: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {}
        for idx, obs in enumerate(rows):
            toks = obs.prompt.split()
            filler = toks[pos]
            start_ids = [n.node_id for n in surface.get(filler, [])]
            target_ids = {n.node_id for n in surface.get(obs.target_surface, [])}
            if not start_ids or not target_ids:
                continue
            signatures = _path_signatures(start_ids, target_ids, edges, policy.max_hops)
            for sig, quality in signatures.items():
                bucket = per_signature.setdefault(sig, {"examples": {}, "sources": set()})
                previous = bucket["examples"].get(idx)
                if previous is None or quality > previous:
                    bucket["examples"][idx] = quality
                bucket["sources"].add(f"{obs.source.source_id}@{obs.source.source_version}")

        if not per_signature:
            continue
        def lineage_components(data: dict[str, Any]) -> list[LineageComponent]:
            return lineage_graph.components(data["sources"])

        ranked = sorted(
            per_signature.items(),
            key=lambda item: (
                -len(item[1]["examples"]),
                -len(lineage_components(item[1])),
                -sum(item[1]["examples"].values()) / max(1, len(item[1]["examples"])),
                item[0],
            ),
        )
        top_support = len(ranked[0][1]["examples"])
        second_support = len(ranked[1][1]["examples"]) if len(ranked) > 1 else 0
        tied_top = sum(1 for _sig, data in ranked if len(data["examples"]) == top_support) > 1

        for sig, data in ranked:
            example_ids = sorted(data["examples"])
            qualities = [float(data["examples"][i]) for i in example_ids]
            support = len(example_ids)
            raw_source_count = len(data["sources"])
            components = lineage_components(data)
            source_count = len(components)
            coverage = support / len(rows)
            margin = support - second_support if support == top_support else support - top_support
            meets_thresholds = (
                support >= policy.min_support
                and source_count >= policy.min_sources
                and coverage + 1e-12 >= policy.min_coverage
                and (sum(qualities) / len(qualities)) + 1e-12 >= policy.min_mean_path_quality
            )
            if support == top_support and tied_top and meets_thresholds:
                status = "contested"
            elif support == top_support and meets_thresholds and margin >= policy.min_support_margin:
                status = "stable"
            else:
                status = "candidate"
            candidate_rows.append(InducedCompositionCandidate(
                pattern_tokens=list(pattern),
                slot_position=pos,
                path=[{"relation_type": rtype, "direction": direction} for rtype, direction in sig],
                support_count=support,
                example_count=len(rows),
                source_count=source_count,
                raw_source_count=raw_source_count,
                distinct_filler_count=len(distinct_fillers),
                coverage=coverage,
                mean_path_quality=sum(qualities) / len(qualities),
                min_path_quality_observed=min(qualities),
                support_margin=margin,
                status=status,
                supporting_prompts=[rows[i].prompt for i in example_ids],
                supporting_targets=[rows[i].target_surface for i in example_ids],
                supporting_sources=sorted(data["sources"]),
                supporting_lineage_components=[list(component.roots) for component in components],
            ))

    # Include token surfaces required by the schema nodes so their member references
    # are valid when this document is compiled independently and then merged.
    schema_nodes: dict[str, IRNode] = {}
    for candidate in candidate_rows:
        for token in candidate.pattern_tokens:
            qs, token_node = make_text_surface(token)
            for q in qs:
                schema_nodes[q.node_id] = q
            schema_nodes[token_node.node_id] = token_node
        node = _schema_node(candidate, policy)
        schema_nodes[node.node_id] = node

    statuses = [x.status for x in candidate_rows]
    return RelationCompositionInductionResult(
        schemas=ForgeDocument(nodes=sorted(schema_nodes.values(), key=lambda n: (n.level, n.node_id)), evidence=[]),
        candidates=candidate_rows,
        group_count=eligible_group_count,
        observation_count=len(observations),
        stable_count=statuses.count("stable"),
        contested_count=statuses.count("contested"),
        candidate_count=statuses.count("candidate"),
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
            if existing is None or (node.metadata and not existing.metadata):
                nodes[node.node_id] = node
        evidence.extend(doc.evidence)
    return ForgeDocument(nodes=sorted(nodes.values(), key=lambda n: (n.level, n.node_id)), evidence=evidence)


def _semantic_fixture(*, competing_path: bool = False, omit_fuchs_path: bool = False) -> ForgeDocument:
    texts = [
        "Hund", "Katze", "Kaninchen", "Fuchs", "Stein", "Haustier", "Wildtier", "Objekt",
        "Haus", "Bau", "Schutz", "Unterschlupf", "wohnt", "in", "[SLOT_0]",
    ]
    nodes: dict[str, IRNode] = {}
    surfaces: dict[str, IRNode] = {}
    for text in texts:
        qs, node = make_text_surface(text)
        for q in qs:
            nodes[q.node_id] = q
        nodes[node.node_id] = node
        surfaces[text] = node

    src = SourceRef("v035-semantic-graph", "1", "composition_induction_fixture")
    evidence: list[RelationEvidence] = []

    def edge(a: str, b: str, rtype: str, score: float = 0.96, confidence: float = 0.97) -> None:
        evidence.append(RelationEvidence(
            surfaces[a].node_id, surfaces[b].node_id, rtype, score, confidence, src,
            {"role": "composition_induction_premise", "predictive_family": False},
        ))

    edge("Hund", "Haustier", RelationType.IS_A.value)
    edge("Katze", "Haustier", RelationType.IS_A.value)
    edge("Kaninchen", "Wildtier", RelationType.IS_A.value)
    if not omit_fuchs_path:
        edge("Fuchs", "Wildtier", RelationType.IS_A.value)
    edge("Stein", "Objekt", RelationType.IS_A.value)
    edge("Haustier", "Haus", RelationType.PROPERTY.value, 0.94, 0.96)
    edge("Wildtier", "Bau", RelationType.PROPERTY.value, 0.93, 0.96)

    if competing_path:
        # Deliberately create a second equally supported program with the same
        # terminals. The inducer must refuse to select one arbitrarily.
        for animal, middle, target in [
            ("Katze", "Schutz", "Haus"),
            ("Kaninchen", "Unterschlupf", "Bau"),
            ("Fuchs", "Unterschlupf", "Bau"),
        ]:
            edge(animal, middle, RelationType.ACTION.value, 0.96, 0.97)
            edge(middle, target, RelationType.PROPERTY.value, 0.94, 0.96)

    return ForgeDocument(nodes=sorted(nodes.values(), key=lambda n: (n.level, n.node_id)), evidence=evidence)


def _observations(*, same_source: bool = False) -> list[RelationCompositionObservation]:
    rows = [
        ("Katze wohnt in", "Haus", "teacher-a"),
        ("Kaninchen wohnt in", "Bau", "teacher-b"),
        ("Fuchs wohnt in", "Bau", "teacher-c"),
    ]
    out = []
    for prompt, target, source_id in rows:
        sid = "teacher-same" if same_source else source_id
        out.append(RelationCompositionObservation(prompt, target, SourceRef(sid, "1", "direct_relation_observation")))
    return out


def _runtime_policy() -> RuntimeCognitionPolicy:
    return RuntimeCognitionPolicy(
        max_recall_nodes=96,
        max_recall_depth=3,
        max_predictions=3,
        min_prediction_score=0.0,
        enable_support_gated_recalled_prediction=True,
        enable_evidence_backed_novel_entity_admission=True,
        novel_entity_min_semantic_quality=0.65,
        enable_relation_composition_prediction=True,
        relation_composition_weight=0.86,
        relation_composition_min_path_quality=0.50,
    )


def _compile_runtime(out: Path, support: ForgeDocument) -> tuple[dict[str, Any], RuntimeCognition]:
    train = [LadderTrainCase(*row) for row in GAUNTLET_DATA["generalization"]["train"]]
    evaluation = [LadderEvalCase(*row) for row in GAUNTLET_DATA["generalization"]["eval"]]
    report = compile_and_benchmark_generalization_ladder(
        ScriptedLadderTeacher(), train, evaluation, out, "v035-ladder", "1",
        probe_cfg=BehaviourProbeConfig(top_k=3, min_probability=0.0, evidence_confidence=0.95),
        cognition_policy=_runtime_policy(), runtime_top_k=3,
        pack_id="v035-relation-composition-induction", pack_version=35,
        support_documents=[support], include_pack_metadata=True,
    )
    return report.to_dict(), RuntimeCognition(load_base_pack(out / "train.eqbase"), _runtime_policy())


def _case(cog: RuntimeCognition, prompt: str) -> dict[str, Any]:
    report = cog.predict(prompt)
    return {
        "prompt": prompt,
        "top1": report.predictions[0].surface if report.predictions else None,
        "top3": [x.surface for x in report.predictions[:3]],
        "composition_schema_checks": int(report.work.get("relation_composition_schema_checks", 0)),
        "composition_matches": int(report.work.get("relation_composition_matches", 0)),
        "composition_admissions": int(report.work.get("relation_composition_admissions", 0)),
        "composition_rejections": int(report.work.get("relation_composition_rejections", 0)),
        "relation_composition": list(report.relation_composition),
    }


def _levels(report: dict[str, Any]) -> dict[int, float]:
    return {int(x["level"]): float(x["top1_retention"]) for x in report["level_reports"]}


def run_relation_composition_induction_benchmark(out_dir: str | Path) -> RelationCompositionInductionBenchmarkReport:
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    policy = RelationCompositionInductionPolicy()
    semantic = _semantic_fixture()
    positive = induce_relation_composition_schemas(semantic, _observations(), policy)
    same_source = induce_relation_composition_schemas(semantic, _observations(same_source=True), policy)
    sparse = induce_relation_composition_schemas(_semantic_fixture(omit_fuchs_path=True), _observations(), policy)
    contested = induce_relation_composition_schemas(_semantic_fixture(competing_path=True), _observations(), policy)

    support = _merge_documents(
        novel_entity_support_document(include_semantics=True),
        semantic,
        positive.schemas,
    )
    ladder_report, cognition = _compile_runtime(out / "runtime", support)
    levels = _levels(ladder_report)

    cases = {
        "heldout_hund": _case(cognition, "Hund wohnt in"),
        "negative_stein": _case(cognition, "Stein wohnt in"),
        "unmatched_phrase": _case(cognition, "Hund wohnt aus"),
    }
    stable_candidates = [x.to_dict() for x in positive.candidates if x.status == "stable"]
    same_source_stable = [x.to_dict() for x in same_source.candidates if x.status == "stable"]
    sparse_stable = [x.to_dict() for x in sparse.candidates if x.status == "stable"]
    contested_stable = [x.to_dict() for x in contested.candidates if x.status == "stable"]

    headline = {
        "l0_exact_replay_top1": levels.get(0, 0.0),
        "l1_suffix_transfer_top1": levels.get(1, 0.0),
        "l2_recombination_top1": levels.get(2, 0.0),
        "l3_novel_order_top1": levels.get(3, 0.0),
        "l4_evidence_backed_novel_entity_top1": levels.get(4, 0.0),
        "l5_auto_induced_relation_composition_top1": levels.get(5, 0.0),
        "stable_schema_count": positive.stable_count,
        "same_source_stable_schema_count": len(same_source_stable),
        "sparse_evidence_stable_schema_count": len(sparse_stable),
        "competing_program_stable_schema_count": len(contested_stable),
        "heldout_hund_is_haus": cases["heldout_hund"]["top1"] == "Haus",
        "stein_abstains": cases["negative_stein"]["top1"] is None,
        "unmatched_phrase_abstains": cases["unmatched_phrase"]["top1"] is None,
    }
    controls = {
        "direct_observations_compiled_into_runtime": False,
        "relation_phrase_supplied_to_inducer": False,
        "heldout_hund_present_in_induction_observations": False,
        "same_source_is_not_independent_consensus": len(same_source_stable) == 0,
        "incomplete_coverage_is_not_stable": len(sparse_stable) == 0,
        "equal_competing_programs_are_not_arbitrarily_selected": len(contested_stable) == 0,
    }
    report = RelationCompositionInductionBenchmarkReport(
        format=RELATION_COMPOSITION_INDUCTION_FORMAT,
        version=RELATION_COMPOSITION_INDUCTION_VERSION,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        headline_metrics=headline,
        induction={
            "positive": positive.to_dict(),
            "same_source_control": same_source.to_dict(),
            "sparse_control": sparse.to_dict(),
            "competing_program_control": contested.to_dict(),
        },
        controls=controls,
        cases=cases,
        notes=[
            "v0.35 discovers the one-slot prompt pattern and semantic path signature; neither is supplied as a composition schema fixture.",
            "Direct induction observations are offline evidence only and are not compiled as prediction edges into the evaluated runtime.",
            "A schema becomes stable only with repeated filler variation, minimum source diversity, sufficient coverage/path quality, and a non-tied winning path signature.",
            "Equal competing path programs remain contested instead of being resolved by arbitrary ordering.",
        ],
    )
    (out / "relation-composition-induction-report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Equinox Forge v0.35 automatic relation-composition induction benchmark")
    ap.add_argument("--out", default="out/v0_35_relation_composition_induction")
    args = ap.parse_args()
    report = run_relation_composition_induction_benchmark(args.out)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
