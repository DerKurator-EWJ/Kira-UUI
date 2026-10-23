from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import sqlite3
import time
from typing import Any

from .composition_lifecycle import (
    CompositionLifecyclePolicy,
    _primary_assessment,
    _runtime_case_for_run,
    apply_composition_lifecycle,
    ensure_composition_lifecycle_schema,
    track_composition_induction_run,
)
from .ir import SourceRef
from .relation_composition_induction import (
    RelationCompositionInductionPolicy,
    RelationCompositionObservation,
    _semantic_fixture,
    induce_relation_composition_schemas,
)
from .source_lineage import (
    SourceLineageGraph,
    ensure_source_lineage_schema,
    register_source_lineage,
)

SOURCE_LINEAGE_BENCHMARK_FORMAT = "equinox-forge-v0.37-source-lineage-correlated-evidence-v1"
SOURCE_LINEAGE_BENCHMARK_VERSION = "0.37"


@dataclass
class SourceLineageBenchmarkReport:
    format: str
    version: str
    elapsed_ms: float
    headline_metrics: dict[str, Any]
    induction_controls: dict[str, Any]
    lifecycle_runs: list[dict[str, Any]]
    controls: dict[str, Any]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _register_parent(db: sqlite3.Connection, child: str, parent: str) -> None:
    register_source_lineage(db, child, "1", parent, "1", relation="derived_from")


def _observations(source_ids: list[str]) -> list[RelationCompositionObservation]:
    if len(source_ids) != 3:
        raise ValueError("fixture expects exactly three source ids")
    rows = [
        ("Katze wohnt in", "Haus", source_ids[0]),
        ("Kaninchen wohnt in", "Bau", source_ids[1]),
        ("Fuchs wohnt in", "Bau", source_ids[2]),
    ]
    return [
        RelationCompositionObservation(prompt, target, SourceRef(source_id, "1", "v037_lineage_fixture"))
        for prompt, target, source_id in rows
    ]


def _primary_candidate(result):
    matches = [
        c for c in result.candidates
        if c.pattern_tokens == ["[SLOT_0]", "wohnt", "in"]
        and [x["relation_type"] for x in c.path] == ["is_a", "property"]
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one primary composition candidate, got {len(matches)}")
    return matches[0]


def run_source_lineage_benchmark(out_dir: str | Path) -> SourceLineageBenchmarkReport:
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    db = sqlite3.connect(out / "source-lineage.sqlite")
    ensure_source_lineage_schema(db)
    ensure_composition_lifecycle_schema(db)

    # Correlated siblings: three apparent sources, one ancestral evidence line.
    for child in ("sib-a", "sib-b", "sib-c"):
        _register_parent(db, child, "foundation-x")

    # Two independent ancestral roots despite three raw sources.
    _register_parent(db, "ind-a", "foundation-x")
    _register_parent(db, "ind-b", "foundation-x")
    _register_parent(db, "ind-c", "foundation-y")

    # Multi-parent bridge conservatively connects both root families into one
    # correlated component instead of pretending the blend is independent.
    _register_parent(db, "blend-xy", "foundation-x")
    _register_parent(db, "blend-xy", "foundation-y")
    _register_parent(db, "x-only", "foundation-x")
    _register_parent(db, "y-only", "foundation-y")

    # Lifecycle support epochs. Epoch 2 only renames descendants of the same roots;
    # epoch 3 introduces a genuinely new root.
    for child, parent in (
        ("s1-a", "support-root-a"), ("s1-b", "support-root-b"), ("s1-c", "support-root-a"),
        ("s2-a", "support-root-a"), ("s2-b", "support-root-b"), ("s2-c", "support-root-a"),
        ("s3-a", "support-root-a"), ("s3-b", "support-root-b"), ("s3-c", "support-root-c"),
        ("c1-a", "conflict-root-a"), ("c1-b", "conflict-root-b"), ("c1-c", "conflict-root-a"),
        ("c2-a", "conflict-root-a"), ("c2-b", "conflict-root-b"), ("c2-c", "conflict-root-a"),
        ("c3-a", "conflict-root-a"), ("c3-b", "conflict-root-b"), ("c3-c", "conflict-root-c"),
    ):
        _register_parent(db, child, parent)

    graph = SourceLineageGraph.from_db(db)
    induction_policy = RelationCompositionInductionPolicy()
    lifecycle_policy = CompositionLifecyclePolicy()

    correlated = induce_relation_composition_schemas(
        _semantic_fixture(), _observations(["sib-a", "sib-b", "sib-c"]), induction_policy, graph
    )
    correlated_primary = _primary_candidate(correlated)

    independent = induce_relation_composition_schemas(
        _semantic_fixture(), _observations(["ind-a", "ind-b", "ind-c"]), induction_policy, graph
    )
    independent_primary = _primary_candidate(independent)

    bridged = induce_relation_composition_schemas(
        _semantic_fixture(), _observations(["blend-xy", "x-only", "y-only"]), induction_policy, graph
    )
    bridged_primary = _primary_candidate(bridged)

    induction_controls = {
        "three_correlated_raw_sources": correlated_primary.to_dict(),
        "two_independent_lineages": independent_primary.to_dict(),
        "multi_parent_bridge": bridged_primary.to_dict(),
    }

    sequence = [
        ("support_1", False, ["s1-a", "s1-b", "s1-c"]),
        ("support_2_same_lineages", False, ["s2-a", "s2-b", "s2-c"]),
        ("support_3_new_lineage", False, ["s3-a", "s3-b", "s3-c"]),
        ("conflict_1", True, ["c1-a", "c1-b", "c1-c"]),
        ("conflict_2_same_lineages", True, ["c2-a", "c2-b", "c2-c"]),
        ("conflict_3_new_lineage", True, ["c3-a", "c3-b", "c3-c"]),
    ]
    expected = ["candidate", "candidate", "stable", "contested", "contested", "quarantined"]
    lifecycle_runs: list[dict[str, Any]] = []
    runtime_top1: list[str | None] = []

    for idx, (name, competing, source_ids) in enumerate(sequence, start=1):
        semantic = _semantic_fixture(competing_path=competing)
        induced = induce_relation_composition_schemas(
            semantic, _observations(source_ids), induction_policy, graph
        )
        lifecycle = track_composition_induction_run(db, induced, lifecycle_policy)
        primary = _primary_assessment(lifecycle)
        governed = apply_composition_lifecycle(induced, lifecycle)
        runtime = _runtime_case_for_run(out / f"runtime_{idx:02d}_{name}", semantic, governed)
        runtime_top1.append(runtime["top1"])
        lifecycle_runs.append({
            "name": name,
            "expected_status": expected[idx - 1],
            "primary": primary,
            "runtime_top1": runtime["top1"],
            "induction_primary": _primary_candidate(induced).to_dict(),
        })

    actual = [row["primary"]["status"] for row in lifecycle_runs]
    controls = {
        "three_sibling_sources_collapse_to_one_lineage": (
            correlated_primary.raw_source_count == 3
            and correlated_primary.source_count == 1
            and correlated_primary.status != "stable"
        ),
        "two_independent_roots_can_satisfy_min_sources": (
            independent_primary.raw_source_count == 3
            and independent_primary.source_count == 2
            and independent_primary.status == "stable"
        ),
        "multi_parent_bridge_is_conservatively_correlated": (
            bridged_primary.raw_source_count == 3
            and bridged_primary.source_count == 1
            and bridged_primary.status != "stable"
        ),
        "same_lineage_support_aliases_do_not_promote": actual[:2] == ["candidate", "candidate"],
        "new_independent_support_lineage_promotes": actual[2] == "stable" and runtime_top1[2] == "Haus",
        "first_conflict_withdraws_runtime": actual[3] == "contested" and runtime_top1[3] is None,
        "same_lineage_conflict_aliases_do_not_quarantine": actual[4] == "contested" and runtime_top1[4] is None,
        "new_independent_conflict_lineage_quarantines": actual[5] == "quarantined" and runtime_top1[5] is None,
        "lifecycle_sequence_matches_contract": actual == expected,
    }

    headline = {
        "raw_correlated_source_count": correlated_primary.raw_source_count,
        "effective_correlated_lineage_count": correlated_primary.source_count,
        "raw_independent_source_count": independent_primary.raw_source_count,
        "effective_independent_lineage_count": independent_primary.source_count,
        "bridged_effective_lineage_count": bridged_primary.source_count,
        "lifecycle_status_sequence": actual,
        "stable_runtime_top1": runtime_top1[2],
        "final_status": actual[-1],
        "all_controls_pass": all(controls.values()),
    }

    report = SourceLineageBenchmarkReport(
        format=SOURCE_LINEAGE_BENCHMARK_FORMAT,
        version=SOURCE_LINEAGE_BENCHMARK_VERSION,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        headline_metrics=headline,
        induction_controls=induction_controls,
        lifecycle_runs=lifecycle_runs,
        controls=controls,
        notes=[
            "v0.37 distinguishes raw source identities from independent ancestral evidence lines.",
            "Sources sharing any transitive ancestry root are conservatively clustered into one correlated lineage component.",
            "A multi-parent derivative bridges its ancestor families, preventing either family from being counted as independent of the blend.",
            "Composition induction uses independent lineage count for min_sources; raw aliases cannot manufacture stable induction.",
            "Composition lifecycle novelty is lineage-root based; renamed descendants of already-seen roots cannot manufacture support or conflict epochs.",
            "Unregistered sources remain backward-compatible and are treated as independent roots of themselves.",
        ],
    )
    (out / "source-lineage-report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    db.close()
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Equinox Forge v0.37 source-lineage / correlated-evidence benchmark")
    ap.add_argument("--out", default="out/v0_37_source_lineage")
    args = ap.parse_args()
    print(json.dumps(run_source_lineage_benchmark(args.out).to_dict(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
