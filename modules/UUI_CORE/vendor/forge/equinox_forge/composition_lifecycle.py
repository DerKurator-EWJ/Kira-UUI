from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import time
from typing import Any

from .ir import ForgeDocument, IRNode, RelationType, SourceRef
from .novel_entity_admission import _support_document as novel_entity_support_document
from .source_lineage import LineageComponent, SourceLineageGraph
from .relation_composition_induction import (
    InducedCompositionCandidate,
    RelationCompositionInductionPolicy,
    RelationCompositionInductionResult,
    RelationCompositionObservation,
    _case,
    _compile_runtime,
    _merge_documents,
    _semantic_fixture,
    induce_relation_composition_schemas,
)

COMPOSITION_LIFECYCLE_FORMAT = "equinox-forge-v0.36-composition-hypothesis-lifecycle-v1"
COMPOSITION_LIFECYCLE_VERSION = "0.36"


@dataclass(frozen=True)
class CompositionLifecyclePolicy:
    """Persistence policy for automatically induced relation-composition programs.

    A single intrinsically strong induction run is not enough for runtime authority.
    Promotion requires repeated clean support epochs with source novelty. Conflict
    epochs likewise require source novelty before they accumulate toward quarantine.
    """

    stable_min_support_epochs: int = 2
    min_sources_per_support_epoch: int = 2
    min_new_support_sources_after_first: int = 1
    quarantine_conflict_epochs: int = 2
    min_new_conflict_sources_after_first: int = 1
    quarantine_after_misses: int = 2

    def validate(self) -> None:
        for name in (
            "stable_min_support_epochs",
            "min_sources_per_support_epoch",
            "min_new_support_sources_after_first",
            "quarantine_conflict_epochs",
            "min_new_conflict_sources_after_first",
            "quarantine_after_misses",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")


@dataclass
class CompositionLifecycleAssessment:
    hypothesis_id: str
    family_id: str
    pattern_tokens: list[str]
    path: list[dict[str, str]]
    induction_status: str
    previous_status: str | None
    status: str
    support_streak: int
    conflict_streak: int
    seen_runs: int
    missed_runs: int
    current_sources: list[str]
    current_lineage_roots: list[str]
    independent_lineage_count: int
    novel_sources_this_epoch: int
    novel_lineage_components_this_epoch: int
    reason: str


@dataclass
class CompositionLifecycleReport:
    format: str
    version: str
    run_id: int
    policy_hash: str
    assessments: list[dict[str, Any]]
    missing_hypotheses: list[dict[str, Any]]
    policy: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CompositionLifecycleBenchmarkReport:
    format: str
    version: str
    elapsed_ms: float
    headline_metrics: dict[str, Any]
    lifecycle_runs: list[dict[str, Any]]
    runtime_cases: dict[str, Any]
    controls: dict[str, Any]
    final_states: list[dict[str, Any]]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ensure_composition_lifecycle_schema(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS composition_hypothesis_runs(
          run_id INTEGER PRIMARY KEY AUTOINCREMENT,
          policy_hash TEXT NOT NULL,
          output_hash TEXT NOT NULL,
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          metadata_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS composition_hypothesis_states(
          hypothesis_id TEXT NOT NULL,
          policy_hash TEXT NOT NULL,
          family_id TEXT NOT NULL,
          pattern_json TEXT NOT NULL,
          path_json TEXT NOT NULL,
          status TEXT NOT NULL,
          first_seen_run INTEGER NOT NULL,
          last_seen_run INTEGER NOT NULL,
          seen_runs INTEGER NOT NULL DEFAULT 1,
          missed_runs INTEGER NOT NULL DEFAULT 0,
          support_streak INTEGER NOT NULL DEFAULT 0,
          conflict_streak INTEGER NOT NULL DEFAULT 0,
          support_sources_seen_json TEXT NOT NULL DEFAULT '[]',
          conflict_sources_seen_json TEXT NOT NULL DEFAULT '[]',
          metadata_json TEXT NOT NULL DEFAULT '{}',
          PRIMARY KEY(hypothesis_id, policy_hash)
        );
        CREATE INDEX IF NOT EXISTS idx_composition_hypothesis_status
          ON composition_hypothesis_states(policy_hash,status,hypothesis_id);
        """
    )


def _stable_hash(payload: Any, prefix: str) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(raw).hexdigest()[:24]}"


def _policy_hash(policy: CompositionLifecyclePolicy) -> str:
    return _stable_hash(asdict(policy), "cpol")


def _family_id(candidate: InducedCompositionCandidate) -> str:
    return _stable_hash(
        {"pattern_tokens": candidate.pattern_tokens, "slot_position": candidate.slot_position},
        "cfam",
    )


def _hypothesis_id(candidate: InducedCompositionCandidate) -> str:
    return _stable_hash(
        {
            "pattern_tokens": candidate.pattern_tokens,
            "slot_position": candidate.slot_position,
            "path": candidate.path,
        },
        "chyp",
    )


def _result_hash(result: RelationCompositionInductionResult) -> str:
    return hashlib.sha256(
        json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _parse_json_list(raw: str | None) -> list[str]:
    try:
        value = json.loads(raw or "[]")
    except Exception:
        return []
    return [str(x) for x in value] if isinstance(value, list) else []


def _load_state_row(db: sqlite3.Connection, hypothesis_id: str, policy_hash: str) -> dict[str, Any] | None:
    row = db.execute(
        """
        SELECT family_id,pattern_json,path_json,status,first_seen_run,last_seen_run,
               seen_runs,missed_runs,support_streak,conflict_streak,
               support_sources_seen_json,conflict_sources_seen_json,metadata_json
        FROM composition_hypothesis_states
        WHERE hypothesis_id=? AND policy_hash=?
        """,
        (hypothesis_id, policy_hash),
    ).fetchone()
    if not row:
        return None
    keys = [
        "family_id", "pattern_json", "path_json", "status", "first_seen_run", "last_seen_run",
        "seen_runs", "missed_runs", "support_streak", "conflict_streak",
        "support_sources_seen_json", "conflict_sources_seen_json", "metadata_json",
    ]
    return dict(zip(keys, row))


def _candidate_lineage_components(candidate: InducedCompositionCandidate) -> list[LineageComponent]:
    raw = getattr(candidate, "supporting_lineage_components", None) or []
    if raw:
        return [LineageComponent(sources=tuple(), roots=tuple(sorted(set(map(str, roots))))) for roots in raw]
    return [LineageComponent(sources=(str(src),), roots=(str(src),)) for src in candidate.supporting_sources]


def track_composition_induction_run(
    db: sqlite3.Connection,
    result: RelationCompositionInductionResult,
    policy: CompositionLifecyclePolicy | None = None,
) -> CompositionLifecycleReport:
    """Persist one composition-induction rebuild and assign runtime-authority status.

    Intrinsic induction status and lifecycle status are intentionally different.
    `stable` from v0.35 means "strong in this rebuild". v0.36 lifecycle `stable`
    means "strong across repeated, source-novel rebuild epochs".
    """

    policy = policy or CompositionLifecyclePolicy()
    policy.validate()
    ensure_composition_lifecycle_schema(db)
    ph = _policy_hash(policy)
    cur = db.execute(
        "INSERT INTO composition_hypothesis_runs(policy_hash,output_hash,metadata_json) VALUES(?,?,?)",
        (ph, _result_hash(result), json.dumps({"version": COMPOSITION_LIFECYCLE_VERSION}, sort_keys=True)),
    )
    run_id = int(cur.lastrowid)

    current_ids = {_hypothesis_id(c) for c in result.candidates}
    missing: list[dict[str, Any]] = []
    for hid, prev_status, missed_runs, support_streak, conflict_streak in db.execute(
        """
        SELECT hypothesis_id,status,missed_runs,support_streak,conflict_streak
        FROM composition_hypothesis_states WHERE policy_hash=?
        """,
        (ph,),
    ).fetchall():
        if hid in current_ids:
            continue
        nm = int(missed_runs) + 1
        status = "quarantined" if nm >= policy.quarantine_after_misses else "contested"
        db.execute(
            """
            UPDATE composition_hypothesis_states
            SET status=?, missed_runs=?, support_streak=0, conflict_streak=0, metadata_json=?
            WHERE hypothesis_id=? AND policy_hash=?
            """,
            (
                status,
                nm,
                json.dumps({"reason": "missing_from_rebuild", "version": COMPOSITION_LIFECYCLE_VERSION}, sort_keys=True),
                hid,
                ph,
            ),
        )
        missing.append(
            {
                "hypothesis_id": hid,
                "previous_status": prev_status,
                "status": status,
                "missed_runs": nm,
                "previous_support_streak": int(support_streak),
                "previous_conflict_streak": int(conflict_streak),
            }
        )

    assessments: list[dict[str, Any]] = []
    for candidate in sorted(result.candidates, key=lambda c: (_family_id(c), _hypothesis_id(c))):
        hid = _hypothesis_id(candidate)
        fid = _family_id(candidate)
        prev = _load_state_row(db, hid, ph)
        previous_status = str(prev["status"]) if prev else None
        first_seen = int(prev["first_seen_run"]) if prev else run_id
        seen_runs = int(prev["seen_runs"]) + 1 if prev else 1
        previous_support_streak = int(prev["support_streak"]) if prev else 0
        previous_conflict_streak = int(prev["conflict_streak"]) if prev else 0
        support_sources_seen = set(_parse_json_list(prev["support_sources_seen_json"])) if prev else set()
        conflict_sources_seen = set(_parse_json_list(prev["conflict_sources_seen_json"])) if prev else set()
        current_sources = set(str(x) for x in candidate.supporting_sources)
        lineage_components = _candidate_lineage_components(candidate)
        current_lineage_roots = set(root for component in lineage_components for root in component.roots)
        independent_lineage_count = len(lineage_components)

        support_streak = previous_support_streak
        conflict_streak = previous_conflict_streak
        novel_sources = 0
        reason = "insufficient_current_evidence"

        if candidate.status == "stable" and independent_lineage_count >= policy.min_sources_per_support_epoch:
            novel_sources = (
                SourceLineageGraph.novel_component_count(lineage_components, support_sources_seen)
                if previous_support_streak > 0 else independent_lineage_count
            )
            source_novel = (
                previous_support_streak == 0
                or novel_sources >= policy.min_new_support_sources_after_first
            )
            if source_novel:
                support_streak = previous_support_streak + 1
                conflict_streak = 0
                status = "stable" if support_streak >= policy.stable_min_support_epochs else "candidate"
                reason = "repeated_source_novel_support" if status == "stable" else "first_clean_support_epoch"
                support_sources_seen |= current_lineage_roots
                conflict_sources_seen = set()
            elif previous_status == "stable":
                # Idempotent rebuilds must not demote an already stable program merely
                # because they contain no new evidence.
                status = "stable"
                conflict_streak = 0
                reason = "stable_idempotent_rebuild"
            else:
                status = "candidate"
                reason = "support_rebuild_without_new_sources"
        elif candidate.status == "contested":
            novel_sources = (
                SourceLineageGraph.novel_component_count(lineage_components, conflict_sources_seen)
                if previous_conflict_streak > 0 else independent_lineage_count
            )
            source_novel = (
                previous_conflict_streak == 0
                or novel_sources >= policy.min_new_conflict_sources_after_first
            )
            support_streak = 0
            if source_novel:
                conflict_streak = previous_conflict_streak + 1
                status = "quarantined" if conflict_streak >= policy.quarantine_conflict_epochs else "contested"
                reason = "repeated_source_novel_conflict" if status == "quarantined" else "first_source_novel_conflict"
                conflict_sources_seen |= current_lineage_roots
                support_sources_seen = set()
            elif previous_status == "quarantined":
                status = "quarantined"
                reason = "quarantined_idempotent_conflict_rebuild"
            else:
                status = "contested"
                reason = "conflict_rebuild_without_new_sources"
        else:
            support_streak = 0
            conflict_streak = 0
            status = "candidate"
            support_sources_seen = set()
            conflict_sources_seen = set()

        metadata = {
            "reason": reason,
            "induction_status": candidate.status,
            "current_sources": sorted(current_sources),
            "current_lineage_roots": sorted(current_lineage_roots),
            "independent_lineage_count": independent_lineage_count,
            "novel_sources_this_epoch": novel_sources,
            "novel_lineage_components_this_epoch": novel_sources,
            "version": COMPOSITION_LIFECYCLE_VERSION,
        }
        db.execute(
            """
            INSERT INTO composition_hypothesis_states(
              hypothesis_id,policy_hash,family_id,pattern_json,path_json,status,
              first_seen_run,last_seen_run,seen_runs,missed_runs,support_streak,conflict_streak,
              support_sources_seen_json,conflict_sources_seen_json,metadata_json
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(hypothesis_id,policy_hash) DO UPDATE SET
              family_id=excluded.family_id,
              pattern_json=excluded.pattern_json,
              path_json=excluded.path_json,
              status=excluded.status,
              last_seen_run=excluded.last_seen_run,
              seen_runs=excluded.seen_runs,
              missed_runs=0,
              support_streak=excluded.support_streak,
              conflict_streak=excluded.conflict_streak,
              support_sources_seen_json=excluded.support_sources_seen_json,
              conflict_sources_seen_json=excluded.conflict_sources_seen_json,
              metadata_json=excluded.metadata_json
            """,
            (
                hid,
                ph,
                fid,
                json.dumps(candidate.pattern_tokens, ensure_ascii=False, sort_keys=True),
                json.dumps(candidate.path, ensure_ascii=False, sort_keys=True),
                status,
                first_seen,
                run_id,
                seen_runs,
                0,
                support_streak,
                conflict_streak,
                json.dumps(sorted(support_sources_seen), ensure_ascii=False),
                json.dumps(sorted(conflict_sources_seen), ensure_ascii=False),
                json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            ),
        )
        assessments.append(
            asdict(
                CompositionLifecycleAssessment(
                    hypothesis_id=hid,
                    family_id=fid,
                    pattern_tokens=list(candidate.pattern_tokens),
                    path=list(candidate.path),
                    induction_status=candidate.status,
                    previous_status=previous_status,
                    status=status,
                    support_streak=support_streak,
                    conflict_streak=conflict_streak,
                    seen_runs=seen_runs,
                    missed_runs=0,
                    current_sources=sorted(current_sources),
                    current_lineage_roots=sorted(current_lineage_roots),
                    independent_lineage_count=independent_lineage_count,
                    novel_sources_this_epoch=novel_sources,
                    novel_lineage_components_this_epoch=novel_sources,
                    reason=reason,
                )
            )
        )

    db.commit()
    return CompositionLifecycleReport(
        format=COMPOSITION_LIFECYCLE_FORMAT,
        version=COMPOSITION_LIFECYCLE_VERSION,
        run_id=run_id,
        policy_hash=ph,
        assessments=assessments,
        missing_hypotheses=missing,
        policy=asdict(policy),
    )


def list_composition_hypothesis_states(db: sqlite3.Connection) -> list[dict[str, Any]]:
    ensure_composition_lifecycle_schema(db)
    rows = db.execute(
        """
        SELECT hypothesis_id,policy_hash,family_id,pattern_json,path_json,status,
               first_seen_run,last_seen_run,seen_runs,missed_runs,support_streak,conflict_streak,
               support_sources_seen_json,conflict_sources_seen_json,metadata_json
        FROM composition_hypothesis_states ORDER BY hypothesis_id,policy_hash
        """
    ).fetchall()
    keys = [
        "hypothesis_id", "policy_hash", "family_id", "pattern_json", "path_json", "status",
        "first_seen_run", "last_seen_run", "seen_runs", "missed_runs", "support_streak", "conflict_streak",
        "support_sources_seen_json", "conflict_sources_seen_json", "metadata_json",
    ]
    out: list[dict[str, Any]] = []
    for row in rows:
        d = dict(zip(keys, row))
        for key in ("pattern_json", "path_json", "support_sources_seen_json", "conflict_sources_seen_json", "metadata_json"):
            d[key.removesuffix("_json") if key.endswith("_json") else key] = json.loads(d.pop(key) or ("[]" if "sources" in key else "{}"))
        out.append(d)
    return out


def apply_composition_lifecycle(
    result: RelationCompositionInductionResult,
    lifecycle: CompositionLifecycleReport,
) -> ForgeDocument:
    """Return schema document whose runtime-visible status is the lifecycle status."""

    doc = deepcopy(result.schemas)
    status_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in lifecycle.assessments:
        pkey = json.dumps(row["pattern_tokens"], ensure_ascii=False, sort_keys=True)
        pathkey = json.dumps(row["path"], ensure_ascii=False, sort_keys=True)
        status_by_key[(pkey, pathkey)] = row

    for node in doc.nodes:
        meta = node.metadata or {}
        if meta.get("role") != "relation_composition_schema":
            continue
        pkey = json.dumps(meta.get("pattern_tokens") or [], ensure_ascii=False, sort_keys=True)
        pathkey = json.dumps(meta.get("path") or [], ensure_ascii=False, sort_keys=True)
        assessment = status_by_key.get((pkey, pathkey))
        if assessment is None:
            meta["induction_status"] = meta.get("status", "candidate")
            meta["status"] = "candidate"
            meta["composition_lifecycle_status"] = "candidate"
            meta["composition_lifecycle_reason"] = "no_lifecycle_assessment"
            continue
        meta["induction_status"] = meta.get("status", assessment["induction_status"])
        meta["status"] = assessment["status"]
        meta["composition_lifecycle_status"] = assessment["status"]
        meta["composition_lifecycle_reason"] = assessment["reason"]
        meta["composition_support_streak"] = assessment["support_streak"]
        meta["composition_conflict_streak"] = assessment["conflict_streak"]
        meta["composition_lifecycle_version"] = COMPOSITION_LIFECYCLE_VERSION
    return doc


def _observations_for_epoch(prefix: str) -> list[RelationCompositionObservation]:
    rows = [
        ("Katze wohnt in", "Haus", f"{prefix}-a"),
        ("Kaninchen wohnt in", "Bau", f"{prefix}-b"),
        ("Fuchs wohnt in", "Bau", f"{prefix}-c"),
    ]
    return [
        RelationCompositionObservation(prompt, target, SourceRef(source_id, "1", "composition_lifecycle_observation"))
        for prompt, target, source_id in rows
    ]


def _primary_assessment(report: CompositionLifecycleReport) -> dict[str, Any]:
    matches = [
        row for row in report.assessments
        if row["pattern_tokens"] == ["[SLOT_0]", "wohnt", "in"]
        and [x["relation_type"] for x in row["path"]] == [RelationType.IS_A.value, RelationType.PROPERTY.value]
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one primary composition hypothesis, got {len(matches)}")
    return matches[0]


def _runtime_case_for_run(out: Path, semantic: ForgeDocument, schemas: ForgeDocument) -> dict[str, Any]:
    support = _merge_documents(novel_entity_support_document(include_semantics=True), semantic, schemas)
    _ladder, cognition = _compile_runtime(out, support)
    return _case(cognition, "Hund wohnt in")


def run_composition_lifecycle_benchmark(out_dir: str | Path) -> CompositionLifecycleBenchmarkReport:
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    induction_policy = RelationCompositionInductionPolicy()
    lifecycle_policy = CompositionLifecyclePolicy()
    db = sqlite3.connect(out / "composition-lifecycle.sqlite")
    ensure_composition_lifecycle_schema(db)

    sequence = [
        ("support_1", _semantic_fixture(), _observations_for_epoch("support-1")),
        ("support_2", _semantic_fixture(), _observations_for_epoch("support-2")),
        ("conflict_1", _semantic_fixture(competing_path=True), _observations_for_epoch("conflict-1")),
        ("conflict_2", _semantic_fixture(competing_path=True), _observations_for_epoch("conflict-2")),
        ("recovery_1", _semantic_fixture(), _observations_for_epoch("recovery-1")),
        ("recovery_2", _semantic_fixture(), _observations_for_epoch("recovery-2")),
    ]

    lifecycle_runs: list[dict[str, Any]] = []
    runtime_cases: dict[str, Any] = {}
    expected_statuses = ["candidate", "stable", "contested", "quarantined", "candidate", "stable"]

    for idx, (name, semantic, observations) in enumerate(sequence, start=1):
        induced = induce_relation_composition_schemas(semantic, observations, induction_policy)
        lifecycle = track_composition_induction_run(db, induced, lifecycle_policy)
        primary = _primary_assessment(lifecycle)
        governed_schemas = apply_composition_lifecycle(induced, lifecycle)
        runtime = _runtime_case_for_run(out / f"runtime_{idx:02d}_{name}", semantic, governed_schemas)
        runtime_cases[name] = runtime
        lifecycle_runs.append(
            {
                "name": name,
                "run_id": lifecycle.run_id,
                "expected_status": expected_statuses[idx - 1],
                "primary": primary,
                "induction": induced.to_dict(),
                "runtime_top1": runtime["top1"],
            }
        )

    final_states = list_composition_hypothesis_states(db)
    db.close()

    # Idempotence controls: replaying the exact same source set must not manufacture
    # an additional epistemic epoch merely because the compiler was rerun.
    idem_db = sqlite3.connect(out / "idempotence-control.sqlite")
    ensure_composition_lifecycle_schema(idem_db)
    idem_support = induce_relation_composition_schemas(
        _semantic_fixture(), _observations_for_epoch("idem-support"), induction_policy
    )
    idem_s1 = track_composition_induction_run(idem_db, idem_support, lifecycle_policy)
    idem_s2 = track_composition_induction_run(idem_db, idem_support, lifecycle_policy)
    idem_support_statuses = [_primary_assessment(idem_s1)["status"], _primary_assessment(idem_s2)["status"]]

    # Establish stable with genuinely different support sources, then replay one
    # identical conflict source set twice. The second replay must not quarantine.
    conflict_db = sqlite3.connect(out / "conflict-idempotence-control.sqlite")
    ensure_composition_lifecycle_schema(conflict_db)
    c_s1 = induce_relation_composition_schemas(_semantic_fixture(), _observations_for_epoch("c-support-1"), induction_policy)
    c_s2 = induce_relation_composition_schemas(_semantic_fixture(), _observations_for_epoch("c-support-2"), induction_policy)
    track_composition_induction_run(conflict_db, c_s1, lifecycle_policy)
    track_composition_induction_run(conflict_db, c_s2, lifecycle_policy)
    same_conflict = induce_relation_composition_schemas(
        _semantic_fixture(competing_path=True), _observations_for_epoch("idem-conflict"), induction_policy
    )
    idem_c1 = track_composition_induction_run(conflict_db, same_conflict, lifecycle_policy)
    idem_c2 = track_composition_induction_run(conflict_db, same_conflict, lifecycle_policy)
    idem_conflict_statuses = [_primary_assessment(idem_c1)["status"], _primary_assessment(idem_c2)["status"]]
    idem_db.close()
    conflict_db.close()

    actual_statuses = [row["primary"]["status"] for row in lifecycle_runs]
    runtime_top1 = [row["runtime_top1"] for row in lifecycle_runs]
    stable_runtime = runtime_top1[1] == "Haus" and runtime_top1[5] == "Haus"
    withdrawn_runtime = all(runtime_top1[i] is None for i in (0, 2, 3, 4))

    controls = {
        "single_clean_run_does_not_authorize_runtime": runtime_top1[0] is None,
        "second_source_novel_support_epoch_promotes": actual_statuses[1] == "stable" and runtime_top1[1] == "Haus",
        "first_conflict_epoch_withdraws_runtime": actual_statuses[2] == "contested" and runtime_top1[2] is None,
        "second_source_novel_conflict_epoch_quarantines": actual_statuses[3] == "quarantined" and runtime_top1[3] is None,
        "first_recovery_epoch_does_not_immediately_restore": actual_statuses[4] == "candidate" and runtime_top1[4] is None,
        "second_source_novel_recovery_epoch_restores": actual_statuses[5] == "stable" and runtime_top1[5] == "Haus",
        "all_lifecycle_transitions_match_contract": actual_statuses == expected_statuses,
        "runtime_authority_tracks_lifecycle_not_single_run_induction": stable_runtime and withdrawn_runtime,
        "identical_support_rebuild_does_not_promote": idem_support_statuses == ["candidate", "candidate"],
        "identical_conflict_rebuild_does_not_quarantine": idem_conflict_statuses == ["contested", "contested"],
    }

    headline = {
        "run_count": len(lifecycle_runs),
        "candidate_after_first_support": actual_statuses[0] == "candidate",
        "stable_after_second_support": actual_statuses[1] == "stable",
        "contested_after_first_conflict": actual_statuses[2] == "contested",
        "quarantined_after_second_conflict": actual_statuses[3] == "quarantined",
        "candidate_after_first_recovery": actual_statuses[4] == "candidate",
        "stable_after_second_recovery": actual_statuses[5] == "stable",
        "stable_runtime_returns_haus": stable_runtime,
        "nonstable_runtime_abstains": withdrawn_runtime,
        "final_primary_status": actual_statuses[-1],
        "idempotent_support_statuses": idem_support_statuses,
        "idempotent_conflict_statuses": idem_conflict_statuses,
    }

    report = CompositionLifecycleBenchmarkReport(
        format=COMPOSITION_LIFECYCLE_FORMAT,
        version=COMPOSITION_LIFECYCLE_VERSION,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        headline_metrics=headline,
        lifecycle_runs=lifecycle_runs,
        runtime_cases=runtime_cases,
        controls=controls,
        final_states=final_states,
        notes=[
            "v0.36 separates single-rebuild induction strength from persistent runtime authority.",
            "A v0.35-stable program enters the persistent lifecycle as candidate and needs a second source-novel clean epoch before runtime activation.",
            "One source-novel conflict epoch withdraws runtime authority immediately; a second source-novel conflict epoch quarantines the program.",
            "Recovery is symmetric: one clean epoch returns only to candidate, a second source-novel clean epoch restores stable runtime authority.",
            "Repeated identical rebuilds do not manufacture promotion or quarantine evidence merely by being rerun.",
        ],
    )
    (out / "composition-lifecycle-report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Equinox Forge v0.36 composition-hypothesis lifecycle benchmark")
    ap.add_argument("--out", default="out/v0_36_composition_lifecycle")
    args = ap.parse_args()
    report = run_composition_lifecycle_benchmark(args.out)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
