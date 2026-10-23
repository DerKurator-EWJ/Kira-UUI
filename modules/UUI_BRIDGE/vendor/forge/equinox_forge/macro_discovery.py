from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable
from collections import defaultdict
import hashlib
import json
import sqlite3

from .macro_builder import MacroCompilePolicy, rebuild_macro, _node

DISCOVERY_FORMAT = "equinox-forge-macro-discovery-v1"
DISCOVERY_VERSION = "0.12"


@dataclass
class MacroDiscoveryPolicy:
    min_pattern_size: int = 2
    max_pattern_size: int = 6
    min_occurrences: int = 2
    min_parent_structures: int = 2
    definition_overhead: float = 1.0
    min_net_savings: float = 1.0
    min_compression_ratio: float = 0.05
    max_candidates: int = 100
    include_canonical: bool = True
    include_derived: bool = True
    derived_statuses: list[str] = field(default_factory=lambda: ["stable", "active"])
    exclude_derived_build_target_prefixes: list[str] = field(default_factory=list)
    prune_dominated: bool = True

    def validate(self) -> None:
        if self.min_pattern_size < 2:
            raise ValueError("min_pattern_size must be >= 2")
        if self.max_pattern_size < self.min_pattern_size:
            raise ValueError("max_pattern_size must be >= min_pattern_size")
        if self.min_occurrences < 2:
            raise ValueError("min_occurrences must be >= 2")
        if self.min_parent_structures < 1:
            raise ValueError("min_parent_structures must be >= 1")
        if self.definition_overhead < 0:
            raise ValueError("definition_overhead must be >= 0")
        if not (0.0 <= self.min_compression_ratio <= 1.0):
            raise ValueError("min_compression_ratio must be in [0,1]")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")
        if not self.include_canonical and not self.include_derived:
            raise ValueError("at least one structure source must be enabled")


@dataclass
class PatternOccurrence:
    parent_id: str
    parent_origin: str
    start: int
    length: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MacroCandidate:
    candidate_id: str
    members: list[str]
    pattern_size: int
    raw_occurrences: int
    usable_occurrences: int
    parent_structures: int
    baseline_refs: float
    encoded_refs: float
    gross_savings: float
    net_savings: float
    compression_ratio: float
    search_steps_saved: float
    eligible: bool
    dominated: bool
    reasons: list[str]
    occurrences: list[PatternOccurrence]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["occurrences"] = [o.to_dict() for o in self.occurrences]
        return d


@dataclass
class MacroDiscoveryReport:
    format: str
    version: str
    policy: dict[str, Any]
    scanned_structures: int
    scanned_member_refs: int
    enumerated_patterns: int
    candidate_count: int
    eligible_count: int
    total_potential_net_savings: float
    candidates: list[MacroCandidate]
    materialized: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "policy": self.policy,
            "scanned_structures": self.scanned_structures,
            "scanned_member_refs": self.scanned_member_refs,
            "enumerated_patterns": self.enumerated_patterns,
            "candidate_count": self.candidate_count,
            "eligible_count": self.eligible_count,
            "total_potential_net_savings": self.total_potential_net_savings,
            "candidates": [c.to_dict() for c in self.candidates],
            "materialized": self.materialized,
        }


def ensure_discovery_schema(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS macro_discovery_runs(
          discovery_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
          policy_hash TEXT NOT NULL,
          scanned_structures INTEGER NOT NULL,
          enumerated_patterns INTEGER NOT NULL,
          eligible_count INTEGER NOT NULL,
          total_potential_net_savings REAL NOT NULL,
          report_json TEXT NOT NULL DEFAULT '{}',
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS macro_candidates(
          discovery_run_id INTEGER NOT NULL,
          candidate_id TEXT NOT NULL,
          members_json TEXT NOT NULL,
          usable_occurrences INTEGER NOT NULL,
          parent_structures INTEGER NOT NULL,
          net_savings REAL NOT NULL,
          compression_ratio REAL NOT NULL,
          eligible INTEGER NOT NULL,
          dominated INTEGER NOT NULL,
          materialized_target_id TEXT,
          metadata_json TEXT NOT NULL DEFAULT '{}',
          PRIMARY KEY(discovery_run_id,candidate_id)
        );
        CREATE INDEX IF NOT EXISTS idx_macro_candidates_candidate ON macro_candidates(candidate_id,discovery_run_id);
        """
    )


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def discovery_policy_hash(policy: MacroDiscoveryPolicy) -> str:
    return hashlib.sha256(_stable_json(asdict(policy)).encode("utf-8")).hexdigest()


def candidate_id_for(members: Iterable[str]) -> str:
    raw = _stable_json(list(members)).encode("utf-8")
    return "mc_" + hashlib.sha256(raw).hexdigest()[:24]


def _structure_sequences(db: sqlite3.Connection, policy: MacroDiscoveryPolicy) -> dict[str, tuple[str, list[str]]]:
    """Return parent_id -> (origin, ordered members), deduplicating canonical/artifact mirrors."""
    out: dict[str, tuple[str, list[str]]] = {}
    if policy.include_canonical:
        rows = db.execute(
            "SELECT n.node_id FROM nodes n WHERE EXISTS(SELECT 1 FROM node_members m WHERE m.parent_id=n.node_id) ORDER BY n.node_id"
        ).fetchall()
        for (parent,) in rows:
            members = [r[0] for r in db.execute(
                "SELECT child_id FROM node_members WHERE parent_id=? ORDER BY position", (parent,)
            ).fetchall()]
            if members:
                out[parent] = ("canonical", members)

    if policy.include_derived:
        placeholders = ",".join("?" for _ in policy.derived_statuses)
        rows = db.execute(
            f"SELECT build_target_id,node_id,payload_json FROM derived_artifacts WHERE node_id IS NOT NULL AND status IN ({placeholders}) ORDER BY build_target_id,artifact_id",
            tuple(policy.derived_statuses),
        ).fetchall() if policy.derived_statuses else []
        for build_target_id, node_id, payload_json in rows:
            if any(build_target_id.startswith(prefix) for prefix in policy.exclude_derived_build_target_prefixes):
                continue
            if node_id in out:
                continue
            try:
                payload = json.loads(payload_json or "{}")
                members = list((payload.get("node") or {}).get("members") or [])
            except Exception:
                members = []
            if members:
                out[node_id] = ("derived", members)
    return out


def _greedy_nonoverlap(starts: list[int], length: int) -> list[int]:
    chosen: list[int] = []
    next_free = -1
    for s in sorted(starts):
        if s >= next_free:
            chosen.append(s)
            next_free = s + length
    return chosen


def _contains_subsequence(big: tuple[str, ...], small: tuple[str, ...]) -> bool:
    if len(small) > len(big):
        return False
    return any(big[i:i+len(small)] == small for i in range(len(big)-len(small)+1))


def discover_macro_candidates(db: sqlite3.Connection, policy: MacroDiscoveryPolicy) -> MacroDiscoveryReport:
    policy.validate()
    structures = _structure_sequences(db, policy)
    raw: dict[tuple[str, ...], list[tuple[str, str, int]]] = defaultdict(list)
    enumerated = 0
    refs = 0
    for parent_id, (origin, members) in structures.items():
        refs += len(members)
        upper = min(policy.max_pattern_size, len(members))
        for size in range(policy.min_pattern_size, upper + 1):
            for start in range(0, len(members)-size+1):
                seq = tuple(members[start:start+size])
                raw[seq].append((parent_id, origin, start))
                enumerated += 1

    candidates: list[MacroCandidate] = []
    for members, occs in raw.items():
        if len(occs) < policy.min_occurrences:
            continue
        size = len(members)
        by_parent: dict[tuple[str, str], list[int]] = defaultdict(list)
        for parent, origin, start in occs:
            by_parent[(parent, origin)].append(start)
        selected: list[PatternOccurrence] = []
        for (parent, origin), starts in sorted(by_parent.items()):
            for start in _greedy_nonoverlap(starts, size):
                selected.append(PatternOccurrence(parent, origin, start, size))
        usable = len(selected)
        parents = len({o.parent_id for o in selected})
        baseline = float(size * usable)
        encoded = float(size + usable)  # one definition + one macro ref per use
        gross = baseline - encoded
        net = gross - float(policy.definition_overhead)
        ratio = (net / baseline) if baseline > 0 else 0.0
        search_saved = float(usable * (size - 1))
        reasons: list[str] = []
        if usable < policy.min_occurrences:
            reasons.append(f"usable_occurrences<{policy.min_occurrences}")
        if parents < policy.min_parent_structures:
            reasons.append(f"parent_structures<{policy.min_parent_structures}")
        if net < policy.min_net_savings:
            reasons.append(f"net_savings<{policy.min_net_savings:g}")
        if ratio < policy.min_compression_ratio:
            reasons.append(f"compression_ratio<{policy.min_compression_ratio:g}")
        candidates.append(MacroCandidate(
            candidate_id_for(members), list(members), size, len(occs), usable, parents,
            baseline, encoded, gross, net, ratio, search_saved,
            not reasons, False, reasons, sorted(selected, key=lambda x:(x.parent_id,x.start)),
        ))

    # Optional dominance pruning: a shorter pattern is not useful if a longer pattern
    # contains it and gives at least as much usable reuse and net compression benefit.
    if policy.prune_dominated:
        eligible = [c for c in candidates if c.eligible]
        for c in eligible:
            small = tuple(c.members)
            for other in eligible:
                if other is c or other.pattern_size <= c.pattern_size:
                    continue
                if other.usable_occurrences < c.usable_occurrences:
                    continue
                if other.net_savings < c.net_savings:
                    continue
                if _contains_subsequence(tuple(other.members), small):
                    c.dominated = True
                    c.eligible = False
                    c.reasons.append(f"dominated_by:{other.candidate_id}")
                    break

    candidates.sort(key=lambda c:(not c.eligible, -c.net_savings, -c.usable_occurrences, -c.pattern_size, c.candidate_id))
    if len(candidates) > policy.max_candidates:
        candidates = candidates[:policy.max_candidates]
    eligible_count = sum(c.eligible for c in candidates)
    total_savings = sum(c.net_savings for c in candidates if c.eligible)
    return MacroDiscoveryReport(
        DISCOVERY_FORMAT, DISCOVERY_VERSION, asdict(policy), len(structures), refs,
        enumerated, len(candidates), eligible_count, total_savings, candidates, [],
    )


def persist_discovery_run(db: sqlite3.Connection, report: MacroDiscoveryReport) -> int:
    ensure_discovery_schema(db)
    ph = hashlib.sha256(_stable_json(report.policy).encode("utf-8")).hexdigest()
    cur = db.execute(
        "INSERT INTO macro_discovery_runs(policy_hash,scanned_structures,enumerated_patterns,eligible_count,total_potential_net_savings,report_json) VALUES(?,?,?,?,?,?)",
        (ph, report.scanned_structures, report.enumerated_patterns, report.eligible_count,
         report.total_potential_net_savings, _stable_json(report.to_dict())),
    )
    run_id = int(cur.lastrowid)
    for c in report.candidates:
        db.execute(
            "INSERT INTO macro_candidates(discovery_run_id,candidate_id,members_json,usable_occurrences,parent_structures,net_savings,compression_ratio,eligible,dominated,metadata_json) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (run_id,c.candidate_id,_stable_json(c.members),c.usable_occurrences,c.parent_structures,
             c.net_savings,c.compression_ratio,int(c.eligible),int(c.dominated),
             _stable_json({"reasons":c.reasons,"occurrences":[o.to_dict() for o in c.occurrences]})),
        )
    db.commit()
    return run_id


def materialize_candidates(
    db: sqlite3.Connection,
    report: MacroDiscoveryReport,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    if limit < 1:
        return []
    out: list[dict[str, Any]] = []
    for c in [x for x in report.candidates if x.eligible][:limit]:
        target_id = f"auto-macro:{c.candidate_id}"
        policy = MacroCompilePolicy(
            input_nodes=list(c.members),
            min_members=len(c.members),
            ordered_members=True,
            surface=f"auto:{c.candidate_id}",
        )
        doc, rep = rebuild_macro(db, target_id, policy, record_store=True)
        out.append({
            "candidate_id": c.candidate_id,
            "target_id": target_id,
            "built": rep.built,
            "output_node_id": rep.output_node_id,
            "level": rep.level,
            "potential_net_savings": c.net_savings,
            "potential_compression_ratio": c.compression_ratio,
            "note": "metrics are potential until parent structures/runtime encoding use the discovered macro",
        })
    report.materialized = out
    return out


def discover_and_optionally_materialize(
    db: sqlite3.Connection,
    policy: MacroDiscoveryPolicy,
    *,
    materialize: bool = False,
    materialize_limit: int = 10,
    persist: bool = True,
) -> tuple[MacroDiscoveryReport, int | None]:
    report = discover_macro_candidates(db, policy)
    if materialize:
        materialize_candidates(db, report, limit=materialize_limit)
    run_id = persist_discovery_run(db, report) if persist else None
    if run_id is not None and report.materialized:
        for m in report.materialized:
            db.execute(
                "UPDATE macro_candidates SET materialized_target_id=? WHERE discovery_run_id=? AND candidate_id=?",
                (m["target_id"], run_id, m["candidate_id"]),
            )
        db.execute(
            "UPDATE macro_discovery_runs SET report_json=? WHERE discovery_run_id=?",
            (_stable_json(report.to_dict()), run_id),
        )
        db.commit()
    return report, run_id
