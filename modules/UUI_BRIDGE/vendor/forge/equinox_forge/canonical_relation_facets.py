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
    ensure_canonical_id_schema,
    identity_record,
    load_numeric_triples,
    register_canonical_identity,
)
from .ir import RelationType
from .macro_numeric_runtime import (
    MacroAwareNumericRuntime,
    MacroNumericRuntimePack,
    MacroNumericResolution,
    MacroShortcut,
    _base_from_payload,
    _digest as _macro_digest,
)

RELATION_FACET_FORMAT = "equinox-canonical-relation-facets-v1"
RELATION_FACET_VERSION = "0.51"
FACET_RUNTIME_FORMAT = "equinox-faceted-macro-numeric-runtime-v1"
FACET_RUNTIME_VERSION = "0.51"


@dataclass(frozen=True)
class RelationFacet:
    facet_predicate_id: int
    family_predicate_id: int
    object_id: int

    def to_list(self) -> list[int]:
        return [int(self.facet_predicate_id), int(self.family_predicate_id), int(self.object_id)]

    @classmethod
    def from_list(cls, row: list[Any]) -> "RelationFacet":
        if len(row) != 3:
            raise ValueError("relation facet row must have three fields")
        return cls(int(row[0]), int(row[1]), int(row[2]))


@dataclass
class RelationFacetBuild:
    format: str
    version: str
    run_id: int
    source_canonical_run_id: int
    derived_canonical_run_id: int
    facet_count: int
    source_triple_count: int
    derived_fact_triple_count: int
    relation_hierarchy_triple_count: int
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FacetedMacroRuntimePack:
    format: str
    version: str
    macro_pack: MacroNumericRuntimePack
    facets: list[RelationFacet]
    digest: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            "format": self.format,
            "version": self.version,
            "macro_pack": self.macro_pack.to_payload(include_digest=True),
            "facets": [f.to_list() for f in self.facets],
            "metadata": self.metadata,
        }
        if include_digest:
            payload["digest"] = self.digest
        return payload


@dataclass(frozen=True)
class FamilyValue:
    family_predicate_id: int
    facet_predicate_id: int
    object_id: int
    route_kind: str
    macro_id: int | None
    origin_subject_id: int | None
    inheritance_depth: int | None
    score: float
    confidence: float
    nodes_touched: int
    baseline_nodes_touched: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FamilyResolution:
    status: str
    subject_id: int
    family_predicate_id: int
    values: tuple[FamilyValue, ...]
    nodes_touched: int
    baseline_nodes_touched: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "subject_id": int(self.subject_id),
            "family_predicate_id": int(self.family_predicate_id),
            "values": [v.to_dict() for v in self.values],
            "nodes_touched": int(self.nodes_touched),
            "baseline_nodes_touched": int(self.baseline_nodes_touched),
        }


def ensure_relation_facet_schema(db: sqlite3.Connection) -> None:
    ensure_canonical_id_schema(db)
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS canonical_relation_facet_runs(
          relation_facet_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
          format TEXT NOT NULL,
          version TEXT NOT NULL,
          source_canonical_run_id INTEGER NOT NULL,
          derived_canonical_run_id INTEGER NOT NULL,
          policy_json TEXT NOT NULL,
          facet_count INTEGER NOT NULL,
          source_triple_count INTEGER NOT NULL,
          derived_fact_triple_count INTEGER NOT NULL,
          relation_hierarchy_triple_count INTEGER NOT NULL,
          digest TEXT NOT NULL,
          metadata_json TEXT NOT NULL DEFAULT '{}',
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS canonical_relation_facets(
          relation_facet_run_id INTEGER NOT NULL,
          facet_predicate_id INTEGER NOT NULL,
          family_predicate_id INTEGER NOT NULL,
          object_id INTEGER NOT NULL,
          PRIMARY KEY(relation_facet_run_id,facet_predicate_id),
          UNIQUE(relation_facet_run_id,family_predicate_id,object_id)
        );
        CREATE INDEX IF NOT EXISTS idx_relation_facets_family
          ON canonical_relation_facets(relation_facet_run_id,family_predicate_id,object_id);
        """
    )


def _stable_digest(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _insert_derived_canonical_run(
    db: sqlite3.Connection,
    *,
    source_run_id: int,
    triples: list[CanonicalTriple],
    metadata: dict[str, Any],
) -> tuple[int, str]:
    identity_count = int(db.execute("SELECT COUNT(*) FROM canonical_identities").fetchone()[0])
    node_count = int(db.execute("SELECT COUNT(*) FROM canonical_identities WHERE namespace='node'").fetchone()[0])
    relation_count = int(db.execute("SELECT COUNT(*) FROM canonical_identities WHERE namespace='relation'").fetchone()[0])
    surface_count = int(db.execute("SELECT COUNT(*) FROM canonical_surface_bindings").fetchone()[0])
    sig = [
        (t.subject_id, t.predicate_id, t.object_id, round(t.score, 12), round(t.confidence, 12), t.source_count, t.evidence_count)
        for t in sorted(triples, key=lambda x: (x.subject_id, x.predicate_id, x.object_id))
    ]
    digest = _stable_digest(sig)
    db.execute(
        """INSERT INTO canonical_id_build_runs(
             format,version,policy_json,identity_count,node_identity_count,
             relation_identity_count,triple_count,surface_binding_count,digest,metadata_json
           ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (
            RELATION_FACET_FORMAT, RELATION_FACET_VERSION,
            json.dumps({"source_canonical_run_id": int(source_run_id)}, sort_keys=True),
            identity_count, node_count, relation_count, len(triples), surface_count, digest,
            json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        ),
    )
    run_id = int(db.execute("SELECT last_insert_rowid()").fetchone()[0])
    for t in triples:
        db.execute(
            """INSERT INTO canonical_relation_triples(
                 canonical_id_run_id,subject_id,predicate_id,object_id,score,confidence,source_count,evidence_count
               ) VALUES(?,?,?,?,?,?,?,?)""",
            (run_id, t.subject_id, t.predicate_id, t.object_id, t.score, t.confidence, t.source_count, t.evidence_count),
        )
    return run_id, digest


def materialize_relation_faceted_id_space(
    db: sqlite3.Connection,
    *,
    source_canonical_run_id: int,
    facet_relation_types: Iterable[str] = (RelationType.PROPERTY.value, RelationType.ACTION.value),
    metadata: dict[str, Any] | None = None,
) -> RelationFacetBuild:
    """Derive first-class relation IDs from coarse relation-family + object identity.

    A fact (subject, PROPERTY, feathers) becomes
        (subject, FACET(PROPERTY, feathers), feathers)
    while the facet relation is registered as a canonical identity and linked to the
    PROPERTY family in canonical_relation_facets. No domain label such as
    'covering' is required. The number itself has no meaning; the relation identity
    is a deterministic handle for the structural pair (family, object).
    """
    ensure_relation_facet_schema(db)
    source_run = int(source_canonical_run_id)
    source = load_numeric_triples(db, source_run)
    facet_family_ids = {canonical_relation_id(db, str(x)) for x in facet_relation_types}
    relation_is_a = canonical_relation_id(db, "relation_is_a")

    facets_by_key: dict[tuple[int, int], RelationFacet] = {}
    derived_facts: list[CanonicalTriple] = []
    for t in source:
        if int(t.predicate_id) not in facet_family_ids:
            derived_facts.append(t)
            continue
        family_record = identity_record(db, int(t.predicate_id))
        symbolic = f"facet:{family_record.symbolic_id}:{int(t.object_id)}"
        facet_id = register_canonical_identity(
            db,
            namespace="relation",
            symbolic_id=symbolic,
            identity_kind="relation_facet",
            metadata={
                "family_predicate_id": int(t.predicate_id),
                "family_symbolic_id": family_record.symbolic_id,
                "object_id": int(t.object_id),
                "builder_version": RELATION_FACET_VERSION,
            },
        )
        facet = RelationFacet(facet_id, int(t.predicate_id), int(t.object_id))
        facets_by_key[(facet.family_predicate_id, facet.object_id)] = facet
        derived_facts.append(CanonicalTriple(
            int(t.subject_id), facet_id, int(t.object_id),
            float(t.score), float(t.confidence), int(t.source_count), int(t.evidence_count),
        ))

    hierarchy: list[CanonicalTriple] = []
    for facet in sorted(facets_by_key.values(), key=lambda x: (x.family_predicate_id, x.object_id, x.facet_predicate_id)):
        # Relation IDs are first-class identities and can themselves participate in
        # the same numeric graph. This edge is metadata/ontology for relation space;
        # entity inheritance still follows only the normal IS_A predicate.
        hierarchy.append(CanonicalTriple(
            facet.facet_predicate_id, relation_is_a, facet.family_predicate_id,
            1.0, 1.0, 1, 1,
        ))

    all_triples = sorted(derived_facts + hierarchy, key=lambda x: (x.subject_id, x.predicate_id, x.object_id))
    derived_run_id, derived_digest = _insert_derived_canonical_run(
        db,
        source_run_id=source_run,
        triples=all_triples,
        metadata={"relation_facets": True, **dict(metadata or {})},
    )
    db.execute(
        """INSERT INTO canonical_relation_facet_runs(
             format,version,source_canonical_run_id,derived_canonical_run_id,policy_json,
             facet_count,source_triple_count,derived_fact_triple_count,relation_hierarchy_triple_count,digest,metadata_json
           ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
        (
            RELATION_FACET_FORMAT, RELATION_FACET_VERSION, source_run, derived_run_id,
            json.dumps({"facet_relation_types": sorted(str(x) for x in facet_relation_types)}, sort_keys=True),
            len(facets_by_key), len(source), len(derived_facts), len(hierarchy), derived_digest,
            json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
        ),
    )
    facet_run_id = int(db.execute("SELECT last_insert_rowid()").fetchone()[0])
    for facet in sorted(facets_by_key.values(), key=lambda x: x.facet_predicate_id):
        db.execute(
            """INSERT INTO canonical_relation_facets(
                 relation_facet_run_id,facet_predicate_id,family_predicate_id,object_id
               ) VALUES(?,?,?,?)""",
            (facet_run_id, facet.facet_predicate_id, facet.family_predicate_id, facet.object_id),
        )
    db.commit()
    return RelationFacetBuild(
        RELATION_FACET_FORMAT, RELATION_FACET_VERSION, facet_run_id, source_run, derived_run_id,
        len(facets_by_key), len(source), len(derived_facts), len(hierarchy), derived_digest,
    )


def load_relation_facets(db: sqlite3.Connection, relation_facet_run_id: int) -> list[RelationFacet]:
    ensure_relation_facet_schema(db)
    rows = db.execute(
        """SELECT facet_predicate_id,family_predicate_id,object_id
           FROM canonical_relation_facets WHERE relation_facet_run_id=?
           ORDER BY family_predicate_id,object_id,facet_predicate_id""",
        (int(relation_facet_run_id),),
    ).fetchall()
    return [RelationFacet(int(a), int(b), int(c)) for a, b, c in rows]


def build_faceted_macro_runtime_pack(
    db: sqlite3.Connection,
    output_path: str | Path,
    *,
    macro_pack: MacroNumericRuntimePack,
    relation_facet_run_id: int,
    metadata: dict[str, Any] | None = None,
) -> FacetedMacroRuntimePack:
    facets = load_relation_facets(db, relation_facet_run_id)
    pack = FacetedMacroRuntimePack(
        FACET_RUNTIME_FORMAT, FACET_RUNTIME_VERSION, macro_pack, facets, "", dict(metadata or {}),
    )
    pack.digest = _stable_digest(pack.to_payload(include_digest=False))
    Path(output_path).write_text(
        json.dumps(pack.to_payload(include_digest=True), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return pack


def _macro_pack_from_payload(payload: dict[str, Any]) -> MacroNumericRuntimePack:
    digest = str(payload.get("digest") or "")
    body = dict(payload)
    body.pop("digest", None)
    if digest != _macro_digest(body):
        raise ValueError("embedded macro numeric runtime pack digest mismatch")
    return MacroNumericRuntimePack(
        str(payload["format"]), str(payload["version"]), _base_from_payload(dict(payload["base"])),
        int(payload["recursive_macro_run_id"]), str(payload["recursive_macro_digest"]),
        [MacroShortcut.from_list(x) for x in payload["shortcuts"]], digest,
        dict(payload.get("metadata") or {}),
    )


def load_faceted_macro_runtime_pack(path: str | Path) -> FacetedMacroRuntimePack:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != FACET_RUNTIME_FORMAT or payload.get("version") != FACET_RUNTIME_VERSION:
        raise ValueError("unsupported faceted macro runtime pack")
    digest = str(payload.get("digest") or "")
    body = dict(payload)
    body.pop("digest", None)
    if digest != _stable_digest(body):
        raise ValueError("faceted macro runtime pack digest mismatch")
    return FacetedMacroRuntimePack(
        str(payload["format"]), str(payload["version"]),
        _macro_pack_from_payload(dict(payload["macro_pack"])),
        [RelationFacet.from_list(x) for x in payload["facets"]], digest,
        dict(payload.get("metadata") or {}),
    )


class FacetedMacroNumericRuntime:
    """DB-free numeric runtime with scalar facet queries and multi-value family views."""

    def __init__(self, pack: FacetedMacroRuntimePack):
        self.pack = pack
        self.macro_runtime = MacroAwareNumericRuntime(pack.macro_pack)
        self._facets_by_family: dict[int, list[RelationFacet]] = {}
        for facet in pack.facets:
            self._facets_by_family.setdefault(int(facet.family_predicate_id), []).append(facet)
        for family in self._facets_by_family:
            self._facets_by_family[family].sort(key=lambda x: (x.object_id, x.facet_predicate_id))

    def resolve(self, subject_id: int, facet_predicate_id: int) -> MacroNumericResolution:
        return self.macro_runtime.resolve(int(subject_id), int(facet_predicate_id))

    def resolve_surface(self, surface: str, *, modality: str = "text", locale: str = "") -> list[int]:
        return self.macro_runtime.base_runtime.resolve_surface(surface, modality=modality, locale=locale)

    def resolve_family(self, subject_id: int, family_predicate_id: int) -> FamilyResolution:
        sid, family = int(subject_id), int(family_predicate_id)
        values: dict[int, FamilyValue] = {}
        touched = 0
        baseline = 0
        for facet in self._facets_by_family.get(family, []):
            result = self.macro_runtime.resolve(sid, facet.facet_predicate_id)
            touched += int(result.nodes_touched)
            baseline += int(result.baseline_nodes_touched)
            r = result.resolution
            if r.status != "resolved" or r.object_id is None or int(r.object_id) != int(facet.object_id):
                continue
            value = FamilyValue(
                family, facet.facet_predicate_id, int(r.object_id), result.route_kind, result.macro_id,
                r.origin_subject_id, r.inheritance_depth, float(r.score), float(r.confidence),
                int(result.nodes_touched), int(result.baseline_nodes_touched),
            )
            incumbent = values.get(value.object_id)
            if incumbent is None:
                values[value.object_id] = value
            else:
                old_depth = 10**9 if incumbent.inheritance_depth is None else int(incumbent.inheritance_depth)
                new_depth = 10**9 if value.inheritance_depth is None else int(value.inheritance_depth)
                if (new_depth, -value.score * value.confidence, value.facet_predicate_id) < (
                    old_depth, -incumbent.score * incumbent.confidence, incumbent.facet_predicate_id
                ):
                    values[value.object_id] = value
        ordered = tuple(sorted(values.values(), key=lambda x: (
            10**9 if x.inheritance_depth is None else x.inheritance_depth, x.object_id, x.facet_predicate_id
        )))
        return FamilyResolution("resolved" if ordered else "not_found", sid, family, ordered, touched, baseline)

    def resolve_surface_family(self, surface: str, family_predicate_id: int, *, modality: str = "text", locale: str = "") -> FamilyResolution:
        ids = self.resolve_surface(surface, modality=modality, locale=locale)
        if len(ids) != 1:
            return FamilyResolution("unknown_or_ambiguous_surface", 0, int(family_predicate_id), (), 0, 0)
        return self.resolve_family(ids[0], family_predicate_id)


def main(argv: list[str] | None = None) -> int:
    from .relation_facet_compiler import run_relation_facet_compiler_trial
    ap = argparse.ArgumentParser(description="Canonical relation facet / multi-valued numeric runtime demo (v0.51).")
    ap.add_argument("--out", default="out/v0_51_relation_facets")
    args = ap.parse_args(argv)
    # The v0.51 end-to-end trial is exposed from teacher_numeric_compiler for one
    # continuous migration path; import lazily to avoid a module cycle at import.
    report = run_relation_facet_compiler_trial(args.out)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
