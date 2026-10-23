from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any
from collections import defaultdict
import hashlib
import json
import sqlite3

from .macro_builder import MacroCompilePolicy, compile_macro, rebuild_macro
from .macro_discovery import (
    MacroCandidate,
    MacroDiscoveryPolicy,
    MacroDiscoveryReport,
    PatternOccurrence,
    discover_macro_candidates,
    _structure_sequences,
)

COMPRESSION_FORMAT = "equinox-forge-runtime-compression-v1"
COMPRESSION_VERSION = "0.13"


@dataclass
class MacroPromotionPolicy:
    min_realized_occurrences: int = 2
    min_realized_parent_structures: int = 2
    definition_overhead: float = 1.0
    min_realized_net_savings: float = 1.0
    min_realized_compression_ratio: float = 0.0
    max_promotions: int = 32

    def validate(self) -> None:
        if self.min_realized_occurrences < 2:
            raise ValueError("min_realized_occurrences must be >= 2")
        if self.min_realized_parent_structures < 1:
            raise ValueError("min_realized_parent_structures must be >= 1")
        if self.definition_overhead < 0:
            raise ValueError("definition_overhead must be >= 0")
        if not (0.0 <= self.min_realized_compression_ratio <= 1.0):
            raise ValueError("min_realized_compression_ratio must be in [0,1]")
        if self.max_promotions < 1:
            raise ValueError("max_promotions must be >= 1")


@dataclass
class PromotedMacro:
    candidate_id: str
    members: list[str]
    target_id: str
    macro_node_id: str
    pattern_size: int
    realized_occurrences: int
    realized_parent_structures: int
    parent_ref_savings: float
    definition_refs: float
    definition_overhead: float
    realized_net_savings: float
    realized_compression_ratio: float
    occurrences: list[PatternOccurrence]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["occurrences"] = [o.to_dict() for o in self.occurrences]
        return d


@dataclass
class ParentRewrite:
    parent_id: str
    parent_origin: str
    original_members: list[str]
    rewritten_members: list[str]
    baseline_refs: int
    rewritten_refs: int
    member_ref_savings: int
    replacements: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeCompressionReport:
    format: str
    version: str
    discovery_policy: dict[str, Any]
    promotion_policy: dict[str, Any]
    scanned_structures: int
    baseline_parent_refs: int
    rewritten_parent_refs: int
    parent_ref_savings: int
    macro_definition_refs: float
    definition_overhead_cost: float
    encoded_total_cost: float
    realized_net_savings: float
    realized_compression_ratio: float
    affected_parents: int
    promoted_count: int
    promoted: list[PromotedMacro]
    rewrites: list[ParentRewrite]
    rejected_after_conflict: list[dict[str, Any]] = field(default_factory=list)
    selection_strategy: str = "deterministic_greedy_marginal_gain"
    compression_run_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "discovery_policy": self.discovery_policy,
            "promotion_policy": self.promotion_policy,
            "scanned_structures": self.scanned_structures,
            "baseline_parent_refs": self.baseline_parent_refs,
            "rewritten_parent_refs": self.rewritten_parent_refs,
            "parent_ref_savings": self.parent_ref_savings,
            "macro_definition_refs": self.macro_definition_refs,
            "definition_overhead_cost": self.definition_overhead_cost,
            "encoded_total_cost": self.encoded_total_cost,
            "realized_net_savings": self.realized_net_savings,
            "realized_compression_ratio": self.realized_compression_ratio,
            "affected_parents": self.affected_parents,
            "promoted_count": self.promoted_count,
            "promoted": [x.to_dict() for x in self.promoted],
            "rewrites": [x.to_dict() for x in self.rewrites],
            "rejected_after_conflict": self.rejected_after_conflict,
            "selection_strategy": self.selection_strategy,
            "compression_run_id": self.compression_run_id,
        }


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def ensure_runtime_compression_schema(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS runtime_compression_runs(
          compression_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
          discovery_policy_hash TEXT NOT NULL,
          promotion_policy_hash TEXT NOT NULL,
          selection_strategy TEXT NOT NULL,
          scanned_structures INTEGER NOT NULL,
          baseline_parent_refs INTEGER NOT NULL,
          rewritten_parent_refs INTEGER NOT NULL,
          parent_ref_savings INTEGER NOT NULL,
          macro_definition_refs REAL NOT NULL,
          definition_overhead_cost REAL NOT NULL,
          encoded_total_cost REAL NOT NULL,
          realized_net_savings REAL NOT NULL,
          realized_compression_ratio REAL NOT NULL,
          affected_parents INTEGER NOT NULL,
          promoted_count INTEGER NOT NULL,
          report_json TEXT NOT NULL DEFAULT '{}',
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS runtime_promoted_macros(
          compression_run_id INTEGER NOT NULL,
          candidate_id TEXT NOT NULL,
          target_id TEXT NOT NULL,
          macro_node_id TEXT NOT NULL,
          members_json TEXT NOT NULL,
          realized_occurrences INTEGER NOT NULL,
          realized_parent_structures INTEGER NOT NULL,
          parent_ref_savings REAL NOT NULL,
          definition_refs REAL NOT NULL,
          definition_overhead REAL NOT NULL,
          realized_net_savings REAL NOT NULL,
          realized_compression_ratio REAL NOT NULL,
          occurrences_json TEXT NOT NULL DEFAULT '[]',
          PRIMARY KEY(compression_run_id,candidate_id)
        );
        CREATE INDEX IF NOT EXISTS idx_runtime_promoted_macro_node
          ON runtime_promoted_macros(macro_node_id,compression_run_id);
        CREATE TABLE IF NOT EXISTS runtime_parent_rewrites(
          compression_run_id INTEGER NOT NULL,
          parent_id TEXT NOT NULL,
          parent_origin TEXT NOT NULL,
          original_members_json TEXT NOT NULL,
          rewritten_members_json TEXT NOT NULL,
          baseline_refs INTEGER NOT NULL,
          rewritten_refs INTEGER NOT NULL,
          member_ref_savings INTEGER NOT NULL,
          replacements_json TEXT NOT NULL DEFAULT '[]',
          PRIMARY KEY(compression_run_id,parent_id)
        );
        CREATE INDEX IF NOT EXISTS idx_runtime_parent_rewrites_parent
          ON runtime_parent_rewrites(parent_id,compression_run_id);
        """
    )


def _policy_hash(obj: Any) -> str:
    return hashlib.sha256(_stable_json(asdict(obj)).encode("utf-8")).hexdigest()


def _interval_free(reserved: dict[str, set[int]], occ: PatternOccurrence) -> bool:
    slots = reserved.get(occ.parent_id, set())
    return all(i not in slots for i in range(occ.start, occ.start + occ.length))


def _available_occurrences(
    candidate: MacroCandidate,
    reserved: dict[str, set[int]],
) -> list[PatternOccurrence]:
    return [o for o in candidate.occurrences if _interval_free(reserved, o)]


def _marginal(candidate: MacroCandidate, occs: list[PatternOccurrence], policy: MacroPromotionPolicy) -> dict[str, Any]:
    uses = len(occs)
    parents = len({o.parent_id for o in occs})
    size = candidate.pattern_size
    parent_savings = float(uses * (size - 1))
    definition_refs = float(size)
    net = parent_savings - definition_refs - float(policy.definition_overhead)
    baseline = float(size * uses)
    ratio = net / baseline if baseline else 0.0
    reasons: list[str] = []
    if uses < policy.min_realized_occurrences:
        reasons.append(f"realized_occurrences<{policy.min_realized_occurrences}")
    if parents < policy.min_realized_parent_structures:
        reasons.append(f"realized_parent_structures<{policy.min_realized_parent_structures}")
    if net < policy.min_realized_net_savings:
        reasons.append(f"realized_net_savings<{policy.min_realized_net_savings:g}")
    if ratio < policy.min_realized_compression_ratio:
        reasons.append(f"realized_compression_ratio<{policy.min_realized_compression_ratio:g}")
    return {
        "candidate": candidate,
        "occurrences": occs,
        "uses": uses,
        "parents": parents,
        "parent_savings": parent_savings,
        "definition_refs": definition_refs,
        "net": net,
        "ratio": ratio,
        "reasons": reasons,
        "eligible": not reasons,
    }


def _select_compatible_candidates(
    candidates: list[MacroCandidate],
    policy: MacroPromotionPolicy,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, set[int]]]:
    reserved: dict[str, set[int]] = defaultdict(set)
    remaining = {c.candidate_id: c for c in candidates if c.eligible}
    selected: list[dict[str, Any]] = []
    rejected: dict[str, dict[str, Any]] = {}

    while remaining and len(selected) < policy.max_promotions:
        evaluated: list[dict[str, Any]] = []
        for c in remaining.values():
            ev = _marginal(c, _available_occurrences(c, reserved), policy)
            evaluated.append(ev)
        viable = [x for x in evaluated if x["eligible"]]
        if not viable:
            for ev in evaluated:
                c = ev["candidate"]
                rejected[c.candidate_id] = {
                    "candidate_id": c.candidate_id,
                    "members": list(c.members),
                    "reasons": ev["reasons"] or ["not_selected"],
                    "remaining_occurrences": ev["uses"],
                    "remaining_parent_structures": ev["parents"],
                    "marginal_net_savings": ev["net"],
                }
            break
        viable.sort(key=lambda x: (
            -x["net"], -x["ratio"], -x["uses"], -x["candidate"].pattern_size,
            x["candidate"].candidate_id,
        ))
        best = viable[0]
        c = best["candidate"]
        selected.append(best)
        for o in best["occurrences"]:
            reserved[o.parent_id].update(range(o.start, o.start + o.length))
        del remaining[c.candidate_id]

        # Record candidates that have become non-viable due to the newly reserved intervals.
        for cid, other in list(remaining.items()):
            ev = _marginal(other, _available_occurrences(other, reserved), policy)
            if not ev["eligible"]:
                rejected[cid] = {
                    "candidate_id": cid,
                    "members": list(other.members),
                    "reasons": ev["reasons"],
                    "remaining_occurrences": ev["uses"],
                    "remaining_parent_structures": ev["parents"],
                    "marginal_net_savings": ev["net"],
                }
                del remaining[cid]

    if remaining:
        for cid, c in sorted(remaining.items()):
            ev = _marginal(c, _available_occurrences(c, reserved), policy)
            rejected[cid] = {
                "candidate_id": cid,
                "members": list(c.members),
                "reasons": [f"max_promotions={policy.max_promotions}"],
                "remaining_occurrences": ev["uses"],
                "remaining_parent_structures": ev["parents"],
                "marginal_net_savings": ev["net"],
            }
    return selected, [rejected[k] for k in sorted(rejected)], reserved


def _materialize_selected(
    db: sqlite3.Connection,
    selected: list[dict[str, Any]],
    policy: MacroPromotionPolicy,
) -> list[PromotedMacro]:
    promoted: list[PromotedMacro] = []
    for ev in selected:
        c: MacroCandidate = ev["candidate"]
        target_id = f"auto-macro:{c.candidate_id}"
        build_policy = MacroCompilePolicy(
            input_nodes=list(c.members),
            min_members=len(c.members),
            ordered_members=True,
            surface=f"auto:{c.candidate_id}",
        )
        _, rep = rebuild_macro(db, target_id, build_policy, record_store=True)
        if not rep.built or not rep.output_node_id:
            raise RuntimeError(f"promotion failed for {c.candidate_id}: {rep.reason or 'unknown'}")
        promoted.append(PromotedMacro(
            candidate_id=c.candidate_id,
            members=list(c.members),
            target_id=target_id,
            macro_node_id=rep.output_node_id,
            pattern_size=c.pattern_size,
            realized_occurrences=ev["uses"],
            realized_parent_structures=ev["parents"],
            parent_ref_savings=ev["parent_savings"],
            definition_refs=ev["definition_refs"],
            definition_overhead=float(policy.definition_overhead),
            realized_net_savings=ev["net"],
            realized_compression_ratio=ev["ratio"],
            occurrences=list(ev["occurrences"]),
        ))
    return promoted


def _rewrite_parents(
    structures: dict[str, tuple[str, list[str]]],
    promoted: list[PromotedMacro],
) -> list[ParentRewrite]:
    replacements_by_parent: dict[str, dict[int, PromotedMacro]] = defaultdict(dict)
    for macro in promoted:
        for o in macro.occurrences:
            if o.start in replacements_by_parent[o.parent_id]:
                raise RuntimeError(f"overlapping replacement start in {o.parent_id}@{o.start}")
            replacements_by_parent[o.parent_id][o.start] = macro

    rewrites: list[ParentRewrite] = []
    for parent_id, (origin, members) in sorted(structures.items()):
        starts = replacements_by_parent.get(parent_id, {})
        rewritten: list[str] = []
        reps: list[dict[str, Any]] = []
        i = 0
        while i < len(members):
            macro = starts.get(i)
            if macro is None:
                rewritten.append(members[i])
                i += 1
                continue
            L = macro.pattern_size
            if members[i:i+L] != macro.members:
                raise RuntimeError(f"rewrite mismatch in {parent_id}@{i} for {macro.candidate_id}")
            rewritten.append(macro.macro_node_id)
            reps.append({
                "start": i,
                "length": L,
                "candidate_id": macro.candidate_id,
                "target_id": macro.target_id,
                "macro_node_id": macro.macro_node_id,
                "replaced_members": list(macro.members),
            })
            i += L
        rewrites.append(ParentRewrite(
            parent_id=parent_id,
            parent_origin=origin,
            original_members=list(members),
            rewritten_members=rewritten,
            baseline_refs=len(members),
            rewritten_refs=len(rewritten),
            member_ref_savings=len(members)-len(rewritten),
            replacements=reps,
        ))
    return rewrites


def _persist_report(db: sqlite3.Connection, report: RuntimeCompressionReport) -> int:
    ensure_runtime_compression_schema(db)
    cur = db.execute(
        "INSERT INTO runtime_compression_runs(discovery_policy_hash,promotion_policy_hash,selection_strategy,scanned_structures,baseline_parent_refs,rewritten_parent_refs,parent_ref_savings,macro_definition_refs,definition_overhead_cost,encoded_total_cost,realized_net_savings,realized_compression_ratio,affected_parents,promoted_count,report_json) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            hashlib.sha256(_stable_json(report.discovery_policy).encode("utf-8")).hexdigest(),
            hashlib.sha256(_stable_json(report.promotion_policy).encode("utf-8")).hexdigest(),
            report.selection_strategy, report.scanned_structures, report.baseline_parent_refs,
            report.rewritten_parent_refs, report.parent_ref_savings, report.macro_definition_refs,
            report.definition_overhead_cost, report.encoded_total_cost, report.realized_net_savings,
            report.realized_compression_ratio, report.affected_parents, report.promoted_count,
            _stable_json(report.to_dict()),
        ),
    )
    run_id = int(cur.lastrowid)
    for m in report.promoted:
        db.execute(
            "INSERT INTO runtime_promoted_macros(compression_run_id,candidate_id,target_id,macro_node_id,members_json,realized_occurrences,realized_parent_structures,parent_ref_savings,definition_refs,definition_overhead,realized_net_savings,realized_compression_ratio,occurrences_json) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                run_id, m.candidate_id, m.target_id, m.macro_node_id, _stable_json(m.members),
                m.realized_occurrences, m.realized_parent_structures, m.parent_ref_savings,
                m.definition_refs, m.definition_overhead, m.realized_net_savings,
                m.realized_compression_ratio, _stable_json([o.to_dict() for o in m.occurrences]),
            ),
        )
    for r in report.rewrites:
        db.execute(
            "INSERT INTO runtime_parent_rewrites(compression_run_id,parent_id,parent_origin,original_members_json,rewritten_members_json,baseline_refs,rewritten_refs,member_ref_savings,replacements_json) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                run_id, r.parent_id, r.parent_origin, _stable_json(r.original_members),
                _stable_json(r.rewritten_members), r.baseline_refs, r.rewritten_refs,
                r.member_ref_savings, _stable_json(r.replacements),
            ),
        )
    report.compression_run_id = run_id
    db.execute(
        "UPDATE runtime_compression_runs SET report_json=? WHERE compression_run_id=?",
        (_stable_json(report.to_dict()), run_id),
    )
    db.commit()
    return run_id


def compile_runtime_compression(
    db: sqlite3.Connection,
    discovery_policy: MacroDiscoveryPolicy,
    promotion_policy: MacroPromotionPolicy,
    *,
    persist: bool = True,
    materialize: bool = True,
) -> RuntimeCompressionReport:
    """Discover, select compatible macros, and produce a physical parent-rewrite view.

    Canonical structures are not mutated. Rewritten member sequences are a runtime encoding
    plan. The selector is deterministic greedy marginal gain; it is deliberately not claimed
    to solve the global weighted packing problem optimally.
    """
    discovery_policy.validate()
    promotion_policy.validate()
    # Previously promoted auto-macro definitions are runtime encodings, not fresh semantic
    # parent observations. Excluding them keeps repeated compilation idempotent and avoids
    # compression plans learning from their own prior output.
    prefixes = list(discovery_policy.exclude_derived_build_target_prefixes)
    if "auto-macro:" not in prefixes:
        prefixes.append("auto-macro:")
    effective_discovery = replace(discovery_policy, exclude_derived_build_target_prefixes=prefixes)
    structures = _structure_sequences(db, effective_discovery)
    discovery: MacroDiscoveryReport = discover_macro_candidates(db, effective_discovery)
    selected, rejected, _ = _select_compatible_candidates(discovery.candidates, promotion_policy)

    if materialize:
        promoted = _materialize_selected(db, selected, promotion_policy)
    else:
        # A rewrite needs macro node ids, but compile_macro is pure with respect to the DB.
        # Use it to derive the exact deterministic ids without registering build targets or
        # writing Derived Artifacts.
        promoted = []
        for ev in selected:
            c: MacroCandidate = ev["candidate"]
            target_id = f"auto-macro:{c.candidate_id}"
            build_policy = MacroCompilePolicy(
                input_nodes=list(c.members), min_members=len(c.members),
                ordered_members=True, surface=f"auto:{c.candidate_id}",
            )
            _, rep = compile_macro(db, target_id, build_policy)
            if not rep.built or not rep.output_node_id:
                raise RuntimeError(f"promotion planning failed for {c.candidate_id}: {rep.reason or 'unknown'}")
            promoted.append(PromotedMacro(
                candidate_id=c.candidate_id, members=list(c.members), target_id=target_id,
                macro_node_id=rep.output_node_id, pattern_size=c.pattern_size,
                realized_occurrences=ev["uses"], realized_parent_structures=ev["parents"],
                parent_ref_savings=ev["parent_savings"], definition_refs=ev["definition_refs"],
                definition_overhead=float(promotion_policy.definition_overhead),
                realized_net_savings=ev["net"], realized_compression_ratio=ev["ratio"],
                occurrences=list(ev["occurrences"]),
            ))

    rewrites = _rewrite_parents(structures, promoted)
    baseline = sum(x.baseline_refs for x in rewrites)
    rewritten = sum(x.rewritten_refs for x in rewrites)
    parent_savings = baseline - rewritten
    definition_refs = sum(x.definition_refs for x in promoted)
    overhead = sum(x.definition_overhead for x in promoted)
    encoded_total = float(rewritten) + definition_refs + overhead
    net = float(baseline) - encoded_total
    ratio = net / float(baseline) if baseline else 0.0
    affected = sum(bool(x.replacements) for x in rewrites)

    # Invariant: global accounting must equal the sum of each selected candidate's
    # marginal contribution because candidate occurrences are non-overlapping.
    marginal_sum = sum(x.realized_net_savings for x in promoted)
    if abs(net - marginal_sum) > 1e-9:
        raise RuntimeError(f"compression accounting mismatch: global={net}, marginal={marginal_sum}")

    report = RuntimeCompressionReport(
        COMPRESSION_FORMAT, COMPRESSION_VERSION,
        asdict(effective_discovery), asdict(promotion_policy), len(structures),
        baseline, rewritten, parent_savings, definition_refs, overhead,
        encoded_total, net, ratio, affected, len(promoted), promoted, rewrites,
        rejected_after_conflict=rejected,
    )
    if persist:
        _persist_report(db, report)
    return report
