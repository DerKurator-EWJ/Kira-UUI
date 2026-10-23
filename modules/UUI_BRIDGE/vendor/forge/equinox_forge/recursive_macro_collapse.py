from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
import argparse
import hashlib
import json
import sqlite3

from .canonical_id_space import (
    canonical_node_id,
    canonical_relation_id,
    latest_run_id as latest_canonical_run_id,
    load_numeric_triples,
    materialize_canonical_id_space,
    register_canonical_identity,
)
from .structural_closure import (
    StructuralClosurePolicy,
    latest_structural_closure_run_id,
    load_materialized_structural_closure,
    materialize_structural_closures,
)
from .ir import IRNode, Modality, NodeKind, RelationEvidence, RelationType, SourceRef

RECURSIVE_MACRO_FORMAT = "equinox-recursive-macro-collapse-v1"
RECURSIVE_MACRO_VERSION = "0.46"
MACRO_MEMBER_RELATION = "macro_member_of"


@dataclass(frozen=True)
class RecursiveMacroPolicy:
    min_members: int = 2
    min_common_facts: int = 2
    max_levels: int = 4
    max_macros_per_level: int = 64
    min_fact_quality: float = 0.0
    min_source_count: int = 1
    include_predicate_types: tuple[str, ...] = ()  # empty => all closure facts
    exclude_predicate_types: tuple[str, ...] = ()
    require_positive_savings: bool = True

    def validate(self) -> None:
        if self.min_members < 2:
            raise ValueError("min_members must be >= 2")
        if self.min_common_facts < 1:
            raise ValueError("min_common_facts must be >= 1")
        if self.max_levels < 1:
            raise ValueError("max_levels must be >= 1")
        if self.max_macros_per_level < 1:
            raise ValueError("max_macros_per_level must be >= 1")
        if not 0.0 <= self.min_fact_quality <= 1.0:
            raise ValueError("min_fact_quality must be in [0,1]")
        if self.min_source_count < 1:
            raise ValueError("min_source_count must be >= 1")


@dataclass(frozen=True)
class MacroFact:
    predicate_id: int
    object_id: int
    score: float
    confidence: float
    support_count: int

    @property
    def quality(self) -> float:
        return max(0.0, min(1.0, float(self.score) * float(self.confidence)))

    def pair(self) -> tuple[int, int]:
        return (self.predicate_id, self.object_id)


@dataclass(frozen=True)
class MacroCandidate:
    member_ids: tuple[int, ...]
    common_pairs: tuple[tuple[int, int], ...]
    potential_savings: int


@dataclass
class RecursiveMacroBuild:
    format: str
    version: str
    run_id: int
    canonical_id_run_id: int
    structural_closure_run_id: int
    levels_built: int
    macro_count: int
    membership_count: int
    local_relation_count: int
    effective_relation_count: int
    potential_reference_savings: int
    digest: str
    policy: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Unit:
    unit_id: int
    kind: str  # concept | macro
    facts: dict[tuple[int, int], MacroFact]


def ensure_recursive_macro_schema(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS recursive_macro_collapse_runs(
          recursive_macro_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
          canonical_id_run_id INTEGER NOT NULL,
          structural_closure_run_id INTEGER NOT NULL,
          format TEXT NOT NULL,
          version TEXT NOT NULL,
          policy_json TEXT NOT NULL,
          levels_built INTEGER NOT NULL,
          macro_count INTEGER NOT NULL,
          membership_count INTEGER NOT NULL,
          local_relation_count INTEGER NOT NULL,
          effective_relation_count INTEGER NOT NULL,
          potential_reference_savings INTEGER NOT NULL,
          digest TEXT NOT NULL,
          metadata_json TEXT NOT NULL DEFAULT '{}',
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS recursive_macros(
          recursive_macro_run_id INTEGER NOT NULL,
          macro_id INTEGER NOT NULL,
          level INTEGER NOT NULL,
          signature_hash TEXT NOT NULL,
          member_count INTEGER NOT NULL,
          effective_fact_count INTEGER NOT NULL,
          local_fact_count INTEGER NOT NULL,
          potential_savings INTEGER NOT NULL,
          status TEXT NOT NULL DEFAULT 'stable',
          metadata_json TEXT NOT NULL DEFAULT '{}',
          PRIMARY KEY(recursive_macro_run_id,macro_id)
        );
        CREATE INDEX IF NOT EXISTS idx_recursive_macro_level
          ON recursive_macros(recursive_macro_run_id,level,macro_id);

        CREATE TABLE IF NOT EXISTS recursive_macro_members(
          recursive_macro_run_id INTEGER NOT NULL,
          macro_id INTEGER NOT NULL,
          member_id INTEGER NOT NULL,
          member_kind TEXT NOT NULL,
          position INTEGER NOT NULL,
          PRIMARY KEY(recursive_macro_run_id,macro_id,member_id)
        );
        CREATE INDEX IF NOT EXISTS idx_recursive_macro_member
          ON recursive_macro_members(recursive_macro_run_id,member_id,macro_id);

        -- Local/residual macro facts only. If a macro later becomes a member of a
        -- higher macro, shared facts are removed here and inherited recursively.
        CREATE TABLE IF NOT EXISTS recursive_macro_relations(
          recursive_macro_run_id INTEGER NOT NULL,
          macro_id INTEGER NOT NULL,
          predicate_id INTEGER NOT NULL,
          object_id INTEGER NOT NULL,
          score REAL NOT NULL,
          confidence REAL NOT NULL,
          support_count INTEGER NOT NULL,
          PRIMARY KEY(recursive_macro_run_id,macro_id,predicate_id,object_id)
        );
        CREATE INDEX IF NOT EXISTS idx_recursive_macro_rel
          ON recursive_macro_relations(recursive_macro_run_id,macro_id,predicate_id,object_id);
        """
    )


def _policy_dict(policy: RecursiveMacroPolicy) -> dict[str, Any]:
    return asdict(policy)


def _signature_hash(pairs: Iterable[tuple[int, int]]) -> str:
    raw = json.dumps([list(x) for x in sorted(set(pairs))], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _macro_id(db: sqlite3.Connection, pairs: Iterable[tuple[int, int]]) -> int:
    sig = _signature_hash(pairs)
    return register_canonical_identity(
        db,
        namespace="macro",
        symbolic_id=f"structural:{sig}",
        identity_kind="recursive_macro",
        metadata={"signature_hash": sig, "builder_version": RECURSIVE_MACRO_VERSION},
    )


def _candidate_groups(units: list[_Unit], policy: RecursiveMacroPolicy) -> list[MacroCandidate]:
    # Pairwise intersections seed frequent structural patterns. Each seed expands
    # to every unit that is a superset of the common pattern. Identical patterns
    # are deduplicated. This is intentionally deterministic and conservative.
    patterns: dict[tuple[tuple[int, int], ...], tuple[int, ...]] = {}
    for i in range(len(units)):
        a = set(units[i].facts)
        for j in range(i + 1, len(units)):
            common = tuple(sorted(a & set(units[j].facts)))
            if len(common) < policy.min_common_facts:
                continue
            members = tuple(sorted(u.unit_id for u in units if set(common).issubset(u.facts)))
            if len(members) < policy.min_members:
                continue
            old = patterns.get(common)
            if old is None or len(members) > len(old):
                patterns[common] = members

    candidates: list[MacroCandidate] = []
    for common, members in patterns.items():
        # Replacing M copies of F shared references with F macro facts + M member
        # links yields this logical hot-path savings. Original base facts remain.
        savings = len(members) * len(common) - (len(common) + len(members))
        if policy.require_positive_savings and savings <= 0:
            continue
        candidates.append(MacroCandidate(members, common, savings))

    # Prefer specific patterns first, then larger support and greater savings.
    candidates.sort(key=lambda c: (-len(c.common_pairs), -len(c.member_ids), -c.potential_savings, c.common_pairs, c.member_ids))
    return candidates


def _select_disjoint(candidates: list[MacroCandidate], policy: RecursiveMacroPolicy) -> list[MacroCandidate]:
    selected: list[MacroCandidate] = []
    used: set[int] = set()
    for c in candidates:
        if any(x in used for x in c.member_ids):
            continue
        selected.append(c)
        used.update(c.member_ids)
        if len(selected) >= policy.max_macros_per_level:
            break
    return selected


def _macro_local_facts(db: sqlite3.Connection, run_id: int, macro_id: int) -> dict[tuple[int, int], MacroFact]:
    rows = db.execute(
        """SELECT predicate_id,object_id,score,confidence,support_count
           FROM recursive_macro_relations
           WHERE recursive_macro_run_id=? AND macro_id=?
           ORDER BY predicate_id,object_id""",
        (int(run_id), int(macro_id)),
    ).fetchall()
    return {
        (int(p), int(o)): MacroFact(int(p), int(o), float(s), float(c), int(sc))
        for p,o,s,c,sc in rows
    }


def effective_macro_facts(
    db: sqlite3.Connection,
    macro_id: int,
    *,
    recursive_macro_run_id: int,
    _seen: set[int] | None = None,
) -> dict[tuple[int, int], MacroFact]:
    """Return a macro's full recursive signature from local residuals + parents."""
    seen = set() if _seen is None else set(_seen)
    mid = int(macro_id)
    if mid in seen:
        return {}
    seen.add(mid)
    out = _macro_local_facts(db, recursive_macro_run_id, mid)
    parents = [int(r[0]) for r in db.execute(
        """SELECT macro_id FROM recursive_macro_members
           WHERE recursive_macro_run_id=? AND member_id=? AND member_kind='macro'
           ORDER BY macro_id""",
        (int(recursive_macro_run_id), mid),
    ).fetchall()]
    for parent in parents:
        for pair, fact in effective_macro_facts(
            db, parent, recursive_macro_run_id=recursive_macro_run_id, _seen=seen
        ).items():
            out.setdefault(pair, fact)
    return out


def _load_concept_units(
    db: sqlite3.Connection,
    *,
    structural_run_id: int,
    root_ids: Iterable[int],
    policy: RecursiveMacroPolicy,
) -> list[_Unit]:
    include_ids = {canonical_relation_id(db, x) for x in policy.include_predicate_types} if policy.include_predicate_types else None
    exclude_ids = {canonical_relation_id(db, x) for x in policy.exclude_predicate_types}
    units: list[_Unit] = []
    for root in sorted({int(x) for x in root_ids}):
        facts: dict[tuple[int, int], MacroFact] = {}
        for f in load_materialized_structural_closure(db, root, structural_run_id=structural_run_id):
            if include_ids is not None and f.predicate_id not in include_ids:
                continue
            if f.predicate_id in exclude_ids:
                continue
            if f.quality + 1e-15 < policy.min_fact_quality:
                continue
            if f.source_count < policy.min_source_count:
                continue
            pair = (f.predicate_id, f.object_id)
            facts[pair] = MacroFact(f.predicate_id, f.object_id, f.score, f.confidence, f.source_count)
        if facts:
            units.append(_Unit(root, "concept", facts))
    return units


def materialize_recursive_macro_collapse(
    db: sqlite3.Connection,
    *,
    canonical_run_id: int | None = None,
    structural_closure_run_id: int | None = None,
    root_ids: Iterable[int] | None = None,
    policy: RecursiveMacroPolicy | None = None,
    metadata: dict[str, Any] | None = None,
) -> RecursiveMacroBuild:
    ensure_recursive_macro_schema(db)
    policy = policy or RecursiveMacroPolicy()
    policy.validate()
    canonical_rid = int(canonical_run_id or latest_canonical_run_id(db))
    structural_rid = int(structural_closure_run_id or latest_structural_closure_run_id(db))

    if root_ids is None:
        roots = [int(r[0]) for r in db.execute(
            """SELECT DISTINCT root_id FROM structural_closure_entries
               WHERE structural_closure_run_id=? ORDER BY root_id""",
            (structural_rid,),
        ).fetchall()]
    else:
        roots = sorted({int(x) for x in root_ids})

    db.execute(
        """INSERT INTO recursive_macro_collapse_runs(
             canonical_id_run_id,structural_closure_run_id,format,version,policy_json,
             levels_built,macro_count,membership_count,local_relation_count,
             effective_relation_count,potential_reference_savings,digest,metadata_json
           ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            canonical_rid, structural_rid, RECURSIVE_MACRO_FORMAT, RECURSIVE_MACRO_VERSION,
            json.dumps(_policy_dict(policy), sort_keys=True, separators=(",", ":")),
            0,0,0,0,0,0,"pending",
            json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
        ),
    )
    run_id = int(db.execute("SELECT last_insert_rowid()").fetchone()[0])

    current = _load_concept_units(
        db, structural_run_id=structural_rid, root_ids=roots, policy=policy
    )
    levels_built = 0
    potential_savings = 0

    for level in range(1, policy.max_levels + 1):
        if len(current) < policy.min_members:
            break
        chosen = _select_disjoint(_candidate_groups(current, policy), policy)
        if not chosen:
            break
        next_units: list[_Unit] = []
        for cand in chosen:
            member_map = {u.unit_id: u for u in current if u.unit_id in set(cand.member_ids)}
            # Conservative fact quality is the weakest supporting member.
            common_facts: dict[tuple[int, int], MacroFact] = {}
            for pair in cand.common_pairs:
                facts = [member_map[mid].facts[pair] for mid in cand.member_ids]
                common_facts[pair] = MacroFact(
                    pair[0], pair[1],
                    min(f.score for f in facts),
                    min(f.confidence for f in facts),
                    len(facts),
                )
            mid = _macro_id(db, common_facts)
            sig = _signature_hash(common_facts)
            db.execute(
                """INSERT OR REPLACE INTO recursive_macros(
                     recursive_macro_run_id,macro_id,level,signature_hash,member_count,
                     effective_fact_count,local_fact_count,potential_savings,status,metadata_json
                   ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id, mid, level, sig, len(cand.member_ids), len(common_facts),
                    len(common_facts), cand.potential_savings, "stable",
                    json.dumps({"member_kinds": sorted({member_map[x].kind for x in cand.member_ids})}, sort_keys=True),
                ),
            )
            for pos, member_id in enumerate(cand.member_ids):
                db.execute(
                    """INSERT OR REPLACE INTO recursive_macro_members(
                         recursive_macro_run_id,macro_id,member_id,member_kind,position
                       ) VALUES(?,?,?,?,?)""",
                    (run_id, mid, member_id, member_map[member_id].kind, pos),
                )
            for fact in common_facts.values():
                db.execute(
                    """INSERT OR REPLACE INTO recursive_macro_relations(
                         recursive_macro_run_id,macro_id,predicate_id,object_id,score,confidence,support_count
                       ) VALUES(?,?,?,?,?,?,?)""",
                    (run_id, mid, fact.predicate_id, fact.object_id, fact.score, fact.confidence, fact.support_count),
                )
            next_units.append(_Unit(mid, "macro", common_facts))
            potential_savings += cand.potential_savings
        if not next_units:
            break
        levels_built = level

        # True recursive collapse in the macro layer: when a macro is a member of
        # a higher macro, facts owned by the parent are removed from the child's
        # local storage. effective_macro_facts() reconstructs them via inheritance.
        if level > 1:
            for parent in next_units:
                parent_pairs = set(parent.facts)
                children = [int(r[0]) for r in db.execute(
                    """SELECT member_id FROM recursive_macro_members
                       WHERE recursive_macro_run_id=? AND macro_id=? AND member_kind='macro'""",
                    (run_id, parent.unit_id),
                ).fetchall()]
                for child in children:
                    for pred, obj in parent_pairs:
                        db.execute(
                            """DELETE FROM recursive_macro_relations
                               WHERE recursive_macro_run_id=? AND macro_id=? AND predicate_id=? AND object_id=?""",
                            (run_id, child, pred, obj),
                        )
        current = next_units

    # Recompute local/effective counts after recursive residual compaction.
    macro_ids = [int(r[0]) for r in db.execute(
        "SELECT macro_id FROM recursive_macros WHERE recursive_macro_run_id=? ORDER BY level,macro_id",
        (run_id,),
    ).fetchall()]
    local_count = int(db.execute(
        "SELECT COUNT(*) FROM recursive_macro_relations WHERE recursive_macro_run_id=?", (run_id,)
    ).fetchone()[0])
    effective_count = 0
    for mid in macro_ids:
        eff = effective_macro_facts(db, mid, recursive_macro_run_id=run_id)
        effective_count += len(eff)
        db.execute(
            """UPDATE recursive_macros SET local_fact_count=?,effective_fact_count=?
               WHERE recursive_macro_run_id=? AND macro_id=?""",
            (len(_macro_local_facts(db, run_id, mid)), len(eff), run_id, mid),
        )
    membership_count = int(db.execute(
        "SELECT COUNT(*) FROM recursive_macro_members WHERE recursive_macro_run_id=?", (run_id,)
    ).fetchone()[0])

    signature = {
        "macros": db.execute(
            """SELECT macro_id,level,signature_hash,member_count,effective_fact_count,local_fact_count
               FROM recursive_macros WHERE recursive_macro_run_id=? ORDER BY level,macro_id""",
            (run_id,),
        ).fetchall(),
        "members": db.execute(
            """SELECT macro_id,member_id,member_kind,position FROM recursive_macro_members
               WHERE recursive_macro_run_id=? ORDER BY macro_id,position,member_id""",
            (run_id,),
        ).fetchall(),
        "relations": db.execute(
            """SELECT macro_id,predicate_id,object_id,score,confidence,support_count
               FROM recursive_macro_relations WHERE recursive_macro_run_id=?
               ORDER BY macro_id,predicate_id,object_id""",
            (run_id,),
        ).fetchall(),
    }
    digest = hashlib.sha256(
        json.dumps(signature, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    db.execute(
        """UPDATE recursive_macro_collapse_runs SET
             levels_built=?,macro_count=?,membership_count=?,local_relation_count=?,
             effective_relation_count=?,potential_reference_savings=?,digest=?
           WHERE recursive_macro_run_id=?""",
        (
            levels_built, len(macro_ids), membership_count, local_count,
            effective_count, potential_savings, digest, run_id,
        ),
    )
    db.commit()
    return RecursiveMacroBuild(
        RECURSIVE_MACRO_FORMAT, RECURSIVE_MACRO_VERSION, run_id, canonical_rid,
        structural_rid, levels_built, len(macro_ids), membership_count, local_count,
        effective_count, potential_savings, digest, _policy_dict(policy),
    )


def latest_recursive_macro_run_id(db: sqlite3.Connection) -> int:
    ensure_recursive_macro_schema(db)
    row = db.execute("SELECT MAX(recursive_macro_run_id) FROM recursive_macro_collapse_runs").fetchone()
    if not row or row[0] is None:
        raise ValueError("no recursive macro collapse build exists")
    return int(row[0])


def macro_members(db: sqlite3.Connection, macro_id: int, *, run_id: int | None = None) -> list[tuple[int, str]]:
    rid = int(run_id or latest_recursive_macro_run_id(db))
    return [(int(a), str(k)) for a,k in db.execute(
        """SELECT member_id,member_kind FROM recursive_macro_members
           WHERE recursive_macro_run_id=? AND macro_id=? ORDER BY position,member_id""",
        (rid, int(macro_id)),
    ).fetchall()]


def _demo_node(name: str) -> IRNode:
    return IRNode(f"concept:{name}", Modality.CROSSMODAL.value, NodeKind.CONCEPT.value, 3, surface=None)


def run_recursive_macro_demo(out_dir: str | Path) -> dict[str, Any]:
    from .database import CanonicalDB

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    db_path = out / "recursive-macro-demo.sqlite"
    if db_path.exists():
        db_path.unlink()

    names = [
        "owl", "falcon", "dog", "wolf", "bird", "mammal", "animal",
        "feathers", "fly", "fur", "walk", "alive", "move",
    ]
    nodes = {x: _demo_node(x) for x in names}
    src = SourceRef("v046-demo", "1", "recursive-macro-collapse")
    ev = [
        RelationEvidence(nodes["owl"].node_id, nodes["bird"].node_id, RelationType.IS_A.value, 1, 1, src),
        RelationEvidence(nodes["falcon"].node_id, nodes["bird"].node_id, RelationType.IS_A.value, 1, 1, src),
        RelationEvidence(nodes["bird"].node_id, nodes["animal"].node_id, RelationType.IS_A.value, 1, 1, src),
        RelationEvidence(nodes["bird"].node_id, nodes["feathers"].node_id, RelationType.PROPERTY.value, .99, .98, src),
        RelationEvidence(nodes["bird"].node_id, nodes["fly"].node_id, RelationType.ACTION.value, .98, .97, src),
        RelationEvidence(nodes["dog"].node_id, nodes["mammal"].node_id, RelationType.IS_A.value, 1, 1, src),
        RelationEvidence(nodes["wolf"].node_id, nodes["mammal"].node_id, RelationType.IS_A.value, 1, 1, src),
        RelationEvidence(nodes["mammal"].node_id, nodes["animal"].node_id, RelationType.IS_A.value, 1, 1, src),
        RelationEvidence(nodes["mammal"].node_id, nodes["fur"].node_id, RelationType.PROPERTY.value, .99, .98, src),
        RelationEvidence(nodes["mammal"].node_id, nodes["walk"].node_id, RelationType.ACTION.value, .98, .97, src),
        RelationEvidence(nodes["animal"].node_id, nodes["alive"].node_id, RelationType.PROPERTY.value, .99, .99, src),
        RelationEvidence(nodes["animal"].node_id, nodes["move"].node_id, RelationType.ACTION.value, .99, .98, src),
    ]

    with CanonicalDB(db_path) as cdb:
        for n in nodes.values():
            cdb.upsert_node(n)
        for x in ev:
            cdb.add_evidence(x)
        cdb.db.commit()
        canonical = materialize_canonical_id_space(cdb.db)
        ids = {name: canonical_node_id(cdb.db, n.node_id) for name,n in nodes.items()}
        base_before = len(load_numeric_triples(cdb.db, canonical.run_id))
        closure = materialize_structural_closures(
            cdb.db,
            canonical_run_id=canonical.run_id,
            root_ids=[ids[x] for x in ("owl", "falcon", "dog", "wolf")],
            policy=StructuralClosurePolicy(max_parent_depth=4),
        )
        build = materialize_recursive_macro_collapse(
            cdb.db,
            canonical_run_id=canonical.run_id,
            structural_closure_run_id=closure.run_id,
            root_ids=[ids[x] for x in ("owl", "falcon", "dog", "wolf")],
            policy=RecursiveMacroPolicy(min_members=2, min_common_facts=2, max_levels=4),
        )
        base_after = len(load_numeric_triples(cdb.db, canonical.run_id))
        macros = [
            {
                "macro_id": int(mid), "level": int(level), "signature_hash": sig,
                "member_count": int(mc), "effective_fact_count": int(ef),
                "local_fact_count": int(lf), "members": macro_members(cdb.db, int(mid), run_id=build.run_id),
                "effective_pairs": [list(k) for k in sorted(effective_macro_facts(cdb.db, int(mid), recursive_macro_run_id=build.run_id))],
                "local_pairs": [list(k) for k in sorted(_macro_local_facts(cdb.db, build.run_id, int(mid)))],
            }
            for mid,level,sig,mc,ef,lf in cdb.db.execute(
                """SELECT macro_id,level,signature_hash,member_count,effective_fact_count,local_fact_count
                   FROM recursive_macros WHERE recursive_macro_run_id=? ORDER BY level,macro_id""",
                (build.run_id,),
            ).fetchall()
        ]
        level1 = [m for m in macros if m["level"] == 1]
        level2 = [m for m in macros if m["level"] == 2]
        macro_ids = {m["macro_id"] for m in macros}
        report = {
            "format": "equinox-forge-recursive-macro-demo-v1",
            "version": RECURSIVE_MACRO_VERSION,
            "canonical_build": canonical.to_dict(),
            "closure_build": closure.to_dict(),
            "macro_build": build.to_dict(),
            "numeric_ids": ids,
            "macros": macros,
            "controls": {
                "canonical_base_unchanged": base_before == base_after == len(ev),
                "two_specific_level1_macros": len(level1) == 2,
                "higher_macro_exists": len(level2) == 1,
                "higher_macro_members_are_macros": (
                    len(level2) == 1 and all(kind == "macro" for _,kind in level2[0]["members"])
                ),
                "macro_ids_are_numeric": all(isinstance(x, int) for x in macro_ids),
                "higher_macro_is_numeric_relation_subject": (
                    len(level2) == 1 and len(level2[0]["effective_pairs"]) >= 2
                ),
                "recursive_residual_compaction_happened": (
                    len(level2) == 1 and all(m["local_fact_count"] < m["effective_fact_count"] for m in level1)
                ),
                "effective_child_signature_survives_compaction": all(len(m["effective_pairs"]) >= 5 for m in level1),
                "positive_reference_savings": build.potential_reference_savings > 0,
            },
        }

    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the Equinox v0.46 recursive numeric macro-collapse demo.")
    ap.add_argument("--out", default="out/v0_46_recursive_macro_collapse")
    args = ap.parse_args(argv)
    report = run_recursive_macro_demo(args.out)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
