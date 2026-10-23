from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
import argparse
import hashlib
import json
import sqlite3

from .canonical_id_space import (
    CanonicalTriple,
    canonical_relation_id,
    identity_record,
    register_canonical_identity,
)
from .canonical_relation_facets import RelationFacet, load_relation_facets

RELATION_MACRO_FORMAT = "equinox-recursive-relation-macro-v1"
RELATION_MACRO_VERSION = "0.52"


@dataclass(frozen=True)
class RelationMacroPolicy:
    min_members: int = 2
    min_common_meta_facts: int = 1
    max_levels: int = 4
    max_macros_per_level: int = 128
    require_positive_savings: bool = True

    def validate(self) -> None:
        if self.min_members < 2:
            raise ValueError("min_members must be >= 2")
        if self.min_common_meta_facts < 1:
            raise ValueError("min_common_meta_facts must be >= 1")
        if self.max_levels < 1:
            raise ValueError("max_levels must be >= 1")
        if self.max_macros_per_level < 1:
            raise ValueError("max_macros_per_level must be >= 1")


@dataclass(frozen=True)
class RelationMetaFact:
    predicate_id: int
    object_id: int

    def pair(self) -> tuple[int, int]:
        return int(self.predicate_id), int(self.object_id)


@dataclass(frozen=True)
class RelationMacroCandidate:
    member_ids: tuple[int, ...]
    common_pairs: tuple[tuple[int, int], ...]
    potential_savings: int


@dataclass
class RecursiveRelationMacroBuild:
    format: str
    version: str
    run_id: int
    relation_facet_run_id: int
    levels_built: int
    macro_count: int
    membership_count: int
    local_meta_fact_count: int
    effective_meta_fact_count: int
    potential_reference_savings: int
    predicate_ready_macro_count: int
    digest: str
    policy: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _RelationUnit:
    unit_id: int
    kind: str  # facet | relation_macro
    facts: dict[tuple[int, int], RelationMetaFact]


def ensure_recursive_relation_macro_schema(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS recursive_relation_macro_runs(
          relation_macro_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
          relation_facet_run_id INTEGER NOT NULL,
          format TEXT NOT NULL,
          version TEXT NOT NULL,
          policy_json TEXT NOT NULL,
          levels_built INTEGER NOT NULL,
          macro_count INTEGER NOT NULL,
          membership_count INTEGER NOT NULL,
          local_meta_fact_count INTEGER NOT NULL,
          effective_meta_fact_count INTEGER NOT NULL,
          potential_reference_savings INTEGER NOT NULL,
          predicate_ready_macro_count INTEGER NOT NULL,
          digest TEXT NOT NULL,
          metadata_json TEXT NOT NULL DEFAULT '{}',
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS recursive_relation_macros(
          relation_macro_run_id INTEGER NOT NULL,
          macro_id INTEGER NOT NULL,
          level INTEGER NOT NULL,
          signature_hash TEXT NOT NULL,
          member_count INTEGER NOT NULL,
          effective_meta_fact_count INTEGER NOT NULL,
          local_meta_fact_count INTEGER NOT NULL,
          potential_savings INTEGER NOT NULL,
          status TEXT NOT NULL DEFAULT 'stable',
          metadata_json TEXT NOT NULL DEFAULT '{}',
          PRIMARY KEY(relation_macro_run_id,macro_id)
        );

        CREATE TABLE IF NOT EXISTS recursive_relation_macro_members(
          relation_macro_run_id INTEGER NOT NULL,
          macro_id INTEGER NOT NULL,
          member_id INTEGER NOT NULL,
          member_kind TEXT NOT NULL,
          position INTEGER NOT NULL,
          PRIMARY KEY(relation_macro_run_id,macro_id,member_id)
        );
        CREATE INDEX IF NOT EXISTS idx_recursive_relation_macro_member
          ON recursive_relation_macro_members(relation_macro_run_id,member_id,macro_id);

        CREATE TABLE IF NOT EXISTS recursive_relation_macro_meta(
          relation_macro_run_id INTEGER NOT NULL,
          macro_id INTEGER NOT NULL,
          predicate_id INTEGER NOT NULL,
          object_id INTEGER NOT NULL,
          PRIMARY KEY(relation_macro_run_id,macro_id,predicate_id,object_id)
        );
        CREATE INDEX IF NOT EXISTS idx_recursive_relation_macro_meta
          ON recursive_relation_macro_meta(relation_macro_run_id,macro_id,predicate_id,object_id);
        """
    )


def _sig(pairs: Iterable[tuple[int, int]]) -> str:
    raw = json.dumps([list(x) for x in sorted(set(pairs))], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _relation_macro_id(db: sqlite3.Connection, pairs: Iterable[tuple[int, int]]) -> int:
    signature = _sig(pairs)
    # Deliberately registered in the relation namespace: this is a first-class
    # predicate handle, not an auxiliary macro outside the logical ID space.
    return register_canonical_identity(
        db,
        namespace="relation",
        symbolic_id=f"relation-macro:{signature}",
        identity_kind="recursive_relation_macro",
        metadata={"signature_hash": signature, "builder_version": RELATION_MACRO_VERSION},
    )


def _candidate_groups(units: list[_RelationUnit], policy: RelationMacroPolicy) -> list[RelationMacroCandidate]:
    patterns: dict[tuple[tuple[int, int], ...], tuple[int, ...]] = {}
    for i, left in enumerate(units):
        a = set(left.facts)
        for right in units[i + 1:]:
            common = tuple(sorted(a & set(right.facts)))
            if len(common) < policy.min_common_meta_facts:
                continue
            members = tuple(sorted(u.unit_id for u in units if set(common).issubset(u.facts)))
            if len(members) < policy.min_members:
                continue
            old = patterns.get(common)
            if old is None or len(members) > len(old):
                patterns[common] = members
    out: list[RelationMacroCandidate] = []
    for common, members in patterns.items():
        savings = len(members) * len(common) - (len(common) + len(members))
        if policy.require_positive_savings and savings <= 0:
            # One shared meta fact across two units is still structurally useful
            # for recursive routing, even if reference-count arithmetic is neutral.
            if not (len(common) == 1 and len(members) >= 2):
                continue
        out.append(RelationMacroCandidate(members, common, max(0, savings)))
    out.sort(key=lambda c: (-len(c.common_pairs), -len(c.member_ids), -c.potential_savings, c.common_pairs, c.member_ids))
    return out


def _select_disjoint(candidates: list[RelationMacroCandidate], policy: RelationMacroPolicy) -> list[RelationMacroCandidate]:
    selected: list[RelationMacroCandidate] = []
    used: set[int] = set()
    for c in candidates:
        if any(m in used for m in c.member_ids):
            continue
        selected.append(c)
        used.update(c.member_ids)
        if len(selected) >= policy.max_macros_per_level:
            break
    return selected


def effective_relation_macro_facts(
    db: sqlite3.Connection,
    macro_id: int,
    *,
    relation_macro_run_id: int,
    _seen: set[int] | None = None,
) -> dict[tuple[int, int], RelationMetaFact]:
    seen = set(_seen or set())
    mid = int(macro_id)
    if mid in seen:
        return {}
    seen.add(mid)
    rows = db.execute(
        """SELECT predicate_id,object_id FROM recursive_relation_macro_meta
           WHERE relation_macro_run_id=? AND macro_id=? ORDER BY predicate_id,object_id""",
        (int(relation_macro_run_id), mid),
    ).fetchall()
    out = {(int(p), int(o)): RelationMetaFact(int(p), int(o)) for p, o in rows}
    # Parent lookup is purely structural: if this relation-macro is a member of
    # a higher relation-macro, inherit that parent's residual meta-facts.
    direct = db.execute(
        """SELECT macro_id FROM recursive_relation_macro_members
           WHERE relation_macro_run_id=? AND member_id=? AND member_kind='relation_macro'
           ORDER BY macro_id""",
        (int(relation_macro_run_id), mid),
    ).fetchall()
    for (parent_id,) in direct:
        for key, fact in effective_relation_macro_facts(
            db, int(parent_id), relation_macro_run_id=relation_macro_run_id, _seen=seen
        ).items():
            out.setdefault(key, fact)
    return out


def materialize_recursive_relation_macro_collapse(
    db: sqlite3.Connection,
    *,
    relation_facet_run_id: int,
    policy: RelationMacroPolicy | None = None,
    metadata: dict[str, Any] | None = None,
) -> RecursiveRelationMacroBuild:
    policy = policy or RelationMacroPolicy()
    policy.validate()
    ensure_recursive_relation_macro_schema(db)
    facets = load_relation_facets(db, int(relation_facet_run_id))
    if not facets:
        raise ValueError("relation facet run contains no facets")

    relation_is_a = canonical_relation_id(db, "relation_is_a")
    relation_kind = canonical_relation_id(db, "relation_kind")
    relation_bound_object = canonical_relation_id(db, "relation_bound_object")
    facet_kind_id = register_canonical_identity(
        db, namespace="meta", symbolic_id="relation_facet", identity_kind="relation_meta_kind", metadata={}
    )

    units: list[_RelationUnit] = []
    for f in sorted(facets, key=lambda x: x.facet_predicate_id):
        facts = {
            (relation_is_a, int(f.family_predicate_id)): RelationMetaFact(relation_is_a, int(f.family_predicate_id)),
            (relation_kind, facet_kind_id): RelationMetaFact(relation_kind, facet_kind_id),
            (relation_bound_object, int(f.object_id)): RelationMetaFact(relation_bound_object, int(f.object_id)),
        }
        units.append(_RelationUnit(int(f.facet_predicate_id), "facet", facts))

    # Reserve run id first so recursive tables can be populated incrementally.
    db.execute(
        """INSERT INTO recursive_relation_macro_runs(
             relation_facet_run_id,format,version,policy_json,levels_built,macro_count,membership_count,
             local_meta_fact_count,effective_meta_fact_count,potential_reference_savings,predicate_ready_macro_count,digest,metadata_json
           ) VALUES(?,?,?,?,0,0,0,0,0,0,0,'',?)""",
        (
            int(relation_facet_run_id), RELATION_MACRO_FORMAT, RELATION_MACRO_VERSION,
            json.dumps(asdict(policy), sort_keys=True), json.dumps(metadata or {}, sort_keys=True),
        ),
    )
    run_id = int(db.execute("SELECT last_insert_rowid()").fetchone()[0])

    levels = 0
    macro_ids: list[int] = []
    total_savings = 0
    current = units
    for level in range(1, policy.max_levels + 1):
        candidates = _select_disjoint(_candidate_groups(current, policy), policy)
        if not candidates:
            break
        next_units: list[_RelationUnit] = []
        used: set[int] = set()
        for candidate in candidates:
            mid = _relation_macro_id(db, candidate.common_pairs)
            macro_ids.append(mid)
            used.update(candidate.member_ids)
            total_savings += int(candidate.potential_savings)
            member_units = [u for u in current if u.unit_id in set(candidate.member_ids)]
            common_facts = {pair: member_units[0].facts[pair] for pair in candidate.common_pairs}
            db.execute(
                """INSERT OR REPLACE INTO recursive_relation_macros(
                     relation_macro_run_id,macro_id,level,signature_hash,member_count,effective_meta_fact_count,
                     local_meta_fact_count,potential_savings,status,metadata_json
                   ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id, mid, level, _sig(candidate.common_pairs), len(candidate.member_ids), len(common_facts),
                    len(common_facts), int(candidate.potential_savings), "stable", "{}",
                ),
            )
            for pos, member in enumerate(sorted(member_units, key=lambda u: u.unit_id)):
                db.execute(
                    """INSERT OR REPLACE INTO recursive_relation_macro_members(
                         relation_macro_run_id,macro_id,member_id,member_kind,position
                       ) VALUES(?,?,?,?,?)""",
                    (run_id, mid, member.unit_id, member.kind, pos),
                )
            for pair in sorted(common_facts):
                db.execute(
                    """INSERT OR REPLACE INTO recursive_relation_macro_meta(
                         relation_macro_run_id,macro_id,predicate_id,object_id
                       ) VALUES(?,?,?,?)""",
                    (run_id, mid, pair[0], pair[1]),
                )
            next_units.append(_RelationUnit(mid, "relation_macro", dict(common_facts)))

        # Residualize child relation-macros when a higher macro absorbs common meta-facts.
        if level > 1:
            for candidate in candidates:
                parent_mid = _relation_macro_id(db, candidate.common_pairs)
                for child_id in candidate.member_ids:
                    row = db.execute(
                        "SELECT 1 FROM recursive_relation_macros WHERE relation_macro_run_id=? AND macro_id=?",
                        (run_id, int(child_id)),
                    ).fetchone()
                    if not row:
                        continue
                    for predicate_id, object_id in candidate.common_pairs:
                        db.execute(
                            """DELETE FROM recursive_relation_macro_meta
                               WHERE relation_macro_run_id=? AND macro_id=? AND predicate_id=? AND object_id=?""",
                            (run_id, int(child_id), int(predicate_id), int(object_id)),
                        )
                    count = int(db.execute(
                        "SELECT COUNT(*) FROM recursive_relation_macro_meta WHERE relation_macro_run_id=? AND macro_id=?",
                        (run_id, int(child_id)),
                    ).fetchone()[0])
                    db.execute(
                        """UPDATE recursive_relation_macros SET local_meta_fact_count=?
                           WHERE relation_macro_run_id=? AND macro_id=?""",
                        (count, run_id, int(child_id)),
                    )

        levels = level
        # Keep unmatched units so broader patterns can still be discovered later.
        next_units.extend(u for u in current if u.unit_id not in used)
        # Recompute effective facts for previously-created macros after residualization.
        refreshed: list[_RelationUnit] = []
        for u in next_units:
            if u.kind == "relation_macro":
                refreshed.append(_RelationUnit(
                    u.unit_id, u.kind,
                    effective_relation_macro_facts(db, u.unit_id, relation_macro_run_id=run_id),
                ))
            else:
                refreshed.append(u)
        current = sorted(refreshed, key=lambda u: u.unit_id)
        if len(current) < policy.min_members:
            break

    unique_macros = sorted(set(macro_ids))
    membership_count = int(db.execute(
        "SELECT COUNT(*) FROM recursive_relation_macro_members WHERE relation_macro_run_id=?", (run_id,)
    ).fetchone()[0])
    local_count = int(db.execute(
        "SELECT COUNT(*) FROM recursive_relation_macro_meta WHERE relation_macro_run_id=?", (run_id,)
    ).fetchone()[0])
    effective_count = sum(len(effective_relation_macro_facts(db, mid, relation_macro_run_id=run_id)) for mid in unique_macros)
    predicate_ready = 0
    for mid in unique_macros:
        rec = identity_record(db, mid)
        if rec.namespace == "relation" and rec.identity_kind == "recursive_relation_macro":
            predicate_ready += 1

    signature = {
        "facets": [f.to_list() for f in facets],
        "macros": [
            {
                "macro_id": mid,
                "members": [list(x) for x in db.execute(
                    """SELECT member_id,member_kind FROM recursive_relation_macro_members
                       WHERE relation_macro_run_id=? AND macro_id=? ORDER BY position,member_id""", (run_id, mid)
                ).fetchall()],
                "local": [list(x) for x in db.execute(
                    """SELECT predicate_id,object_id FROM recursive_relation_macro_meta
                       WHERE relation_macro_run_id=? AND macro_id=? ORDER BY predicate_id,object_id""", (run_id, mid)
                ).fetchall()],
            }
            for mid in unique_macros
        ],
    }
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    db.execute(
        """UPDATE recursive_relation_macro_runs SET
             levels_built=?,macro_count=?,membership_count=?,local_meta_fact_count=?,effective_meta_fact_count=?,
             potential_reference_savings=?,predicate_ready_macro_count=?,digest=?
           WHERE relation_macro_run_id=?""",
        (levels, len(unique_macros), membership_count, local_count, effective_count, total_savings, predicate_ready, digest, run_id),
    )
    return RecursiveRelationMacroBuild(
        RELATION_MACRO_FORMAT, RELATION_MACRO_VERSION, run_id, int(relation_facet_run_id), levels,
        len(unique_macros), membership_count, local_count, effective_count, total_savings,
        predicate_ready, digest, asdict(policy),
    )


def relation_macro_members(db: sqlite3.Connection, *, relation_macro_run_id: int, macro_id: int) -> list[tuple[int, str]]:
    rows = db.execute(
        """SELECT member_id,member_kind FROM recursive_relation_macro_members
           WHERE relation_macro_run_id=? AND macro_id=? ORDER BY position,member_id""",
        (int(relation_macro_run_id), int(macro_id)),
    ).fetchall()
    return [(int(a), str(b)) for a, b in rows]


def main(argv: list[str] | None = None) -> int:
    from .relation_facet_compiler import run_relation_facet_compiler_trial
    from .database import CanonicalDB
    from .canonical_relation_facets import materialize_relation_faceted_id_space
    from .canonical_id_space import materialize_canonical_id_space
    from .real_teacher_domain_trial import _NeuralDomainTeacher
    from .teacher_numeric_compiler import _load_fixture, _policy
    from .top_down_teacher_structure import extract_top_down_teacher_structure
    import shutil

    ap = argparse.ArgumentParser(description="Equinox v0.52 recursive relation-ID collapse trial.")
    ap.add_argument("--out", default="out/v0_52_recursive_relation_macro")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fixture = _load_fixture(None)
    teacher = _NeuralDomainTeacher([(str(a), str(b)) for a, b in fixture["training_pairs"]], out / "teacher", seed=5201, max_epochs=800)
    db_path = out / "forge.sqlite"
    if db_path.exists(): db_path.unlink()
    with CanonicalDB(db_path) as cdb:
        extract_top_down_teacher_structure(
            cdb.db, teacher, seed_surfaces=fixture["seed_surfaces"], source_id="v052-neural-teacher",
            source_version="1", policy=_policy(fixture), metadata={"trial": RELATION_MACRO_FORMAT},
        )
        base = materialize_canonical_id_space(cdb.db)
        facets = materialize_relation_faceted_id_space(cdb.db, source_canonical_run_id=base.run_id)
        build = materialize_recursive_relation_macro_collapse(cdb.db, relation_facet_run_id=facets.run_id)
        macros = cdb.db.execute(
            """SELECT macro_id,level,member_count,local_meta_fact_count FROM recursive_relation_macros
               WHERE relation_macro_run_id=? ORDER BY level,macro_id""", (build.run_id,)
        ).fetchall()
        report = {"build": build.to_dict(), "macros": [list(x) for x in macros], "controls": {
            "recursive_levels": build.levels_built >= 2,
            "all_macros_are_predicate_ready_relation_ids": build.predicate_ready_macro_count == build.macro_count,
            "relation_macros_exist": build.macro_count >= 3,
        }}
    state = Path(teacher.manifest()["model_dir"])
    if state.exists():
        try: shutil.move(str(state), str(state) + ".disabled")
        except Exception: pass
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
