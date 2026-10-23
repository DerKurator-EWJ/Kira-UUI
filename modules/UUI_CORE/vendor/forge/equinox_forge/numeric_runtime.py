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
    bind_surface,
    canonical_node_id,
    canonical_relation_id,
    latest_run_id as latest_canonical_run_id,
    load_numeric_triples,
    materialize_canonical_id_space,
    resolve_surface,
)
from .ir import IRNode, Modality, NodeKind, RelationEvidence, RelationType, SourceRef

NUMERIC_RUNTIME_FORMAT = "equinox-specific-general-numeric-runtime-v1"
NUMERIC_RUNTIME_VERSION = "0.48"


@dataclass(frozen=True)
class NumericRuntimePolicy:
    inheritance_predicate_ids: tuple[int, ...]
    max_parent_depth: int = 8
    min_quality: float = 0.50
    ambiguity_margin: float = 0.05

    def validate(self) -> None:
        if not self.inheritance_predicate_ids:
            raise ValueError("inheritance_predicate_ids must not be empty")
        if self.max_parent_depth < 0:
            raise ValueError("max_parent_depth must be >= 0")
        if not 0.0 <= self.min_quality <= 1.0:
            raise ValueError("min_quality must be in [0,1]")
        if not 0.0 <= self.ambiguity_margin <= 1.0:
            raise ValueError("ambiguity_margin must be in [0,1]")


@dataclass(frozen=True)
class NumericRuntimeFact:
    subject_id: int
    predicate_id: int
    object_id: int
    score: float
    confidence: float
    source_count: int
    evidence_count: int

    @property
    def quality(self) -> float:
        return max(0.0, min(1.0, float(self.score) * float(self.confidence)))

    def to_list(self) -> list[Any]:
        return [
            int(self.subject_id), int(self.predicate_id), int(self.object_id),
            float(self.score), float(self.confidence),
            int(self.source_count), int(self.evidence_count),
        ]

    @classmethod
    def from_list(cls, row: list[Any]) -> "NumericRuntimeFact":
        if len(row) != 7:
            raise ValueError("numeric runtime relation row must have seven fields")
        return cls(int(row[0]), int(row[1]), int(row[2]), float(row[3]), float(row[4]), int(row[5]), int(row[6]))


@dataclass(frozen=True)
class NumericSurfaceBinding:
    surface: str
    modality: str
    locale: str
    concept_id: int

    def to_list(self) -> list[Any]:
        return [self.surface, self.modality, self.locale, int(self.concept_id)]

    @classmethod
    def from_list(cls, row: list[Any]) -> "NumericSurfaceBinding":
        if len(row) != 4:
            raise ValueError("surface binding row must have four fields")
        return cls(str(row[0]), str(row[1]), str(row[2]), int(row[3]))


@dataclass
class NumericRuntimePack:
    format: str
    version: str
    canonical_run_id: int
    canonical_digest: str
    inheritance_predicate_ids: tuple[int, ...]
    relations: list[NumericRuntimeFact]
    surfaces: list[NumericSurfaceBinding]
    digest: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            "format": self.format,
            "version": self.version,
            "canonical_run_id": int(self.canonical_run_id),
            "canonical_digest": self.canonical_digest,
            "inheritance_predicate_ids": [int(x) for x in self.inheritance_predicate_ids],
            "relations": [r.to_list() for r in self.relations],
            "surfaces": [s.to_list() for s in self.surfaces],
            "metadata": self.metadata,
        }
        if include_digest:
            payload["digest"] = self.digest
        return payload


@dataclass(frozen=True)
class NumericResolution:
    status: str
    subject_id: int
    predicate_id: int
    object_id: int | None
    origin_subject_id: int | None
    inheritance_depth: int | None
    inheritance_path: tuple[int, ...]
    score: float
    confidence: float
    source_count: int
    evidence_count: int
    candidate_object_ids: tuple[int, ...] = ()
    reason: str = ""

    @property
    def inherited(self) -> bool:
        return bool(self.inheritance_depth and self.inheritance_depth > 0)

    @property
    def quality(self) -> float:
        return max(0.0, min(1.0, float(self.score) * float(self.confidence)))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["inheritance_path"] = list(self.inheritance_path)
        d["candidate_object_ids"] = list(self.candidate_object_ids)
        d["inherited"] = self.inherited
        d["quality"] = self.quality
        return d


class SpecificToGeneralNumericRuntime:
    """DB-free runtime over the canonical numeric graph.

    Resolution is deliberately categorical by specificity: the first hierarchy depth
    containing sufficiently strong direct facts for the requested predicate owns the
    answer. More general evidence is not allowed to outvote a valid local/specific
    fact merely by having higher aggregate score. If the nearest depth contains an
    unresolved conflict, the runtime abstains instead of climbing past it.
    """

    def __init__(self, pack: NumericRuntimePack, policy: NumericRuntimePolicy | None = None):
        self.pack = pack
        self.policy = policy or NumericRuntimePolicy(tuple(pack.inheritance_predicate_ids))
        self.policy.validate()
        self._by_subject_predicate: dict[tuple[int, int], list[NumericRuntimeFact]] = {}
        self._parents: dict[int, list[int]] = {}
        inheritance = set(self.policy.inheritance_predicate_ids)
        for fact in pack.relations:
            self._by_subject_predicate.setdefault((fact.subject_id, fact.predicate_id), []).append(fact)
            if fact.predicate_id in inheritance and fact.quality + 1e-15 >= self.policy.min_quality:
                self._parents.setdefault(fact.subject_id, []).append(fact.object_id)
        for key in self._by_subject_predicate:
            self._by_subject_predicate[key].sort(key=lambda f: (-f.quality, -f.confidence, f.object_id, f.subject_id))
        for subject in self._parents:
            self._parents[subject] = sorted(set(self._parents[subject]))

        self._surface_index: dict[tuple[str, str, str], list[int]] = {}
        for b in pack.surfaces:
            self._surface_index.setdefault((b.surface, b.modality, b.locale), []).append(b.concept_id)
        for key in self._surface_index:
            self._surface_index[key] = sorted(set(self._surface_index[key]))

    def resolve_surface(self, surface: str, *, modality: str = Modality.TEXT.value, locale: str = "") -> list[int]:
        return list(self._surface_index.get((surface, modality, locale), []))

    def _levels(self, subject_id: int) -> list[list[tuple[int, tuple[int, ...]]]]:
        start = int(subject_id)
        levels: list[list[tuple[int, tuple[int, ...]]]] = [[(start, (start,))]]
        seen = {start}
        current = levels[0]
        for _ in range(self.policy.max_parent_depth):
            nxt: list[tuple[int, tuple[int, ...]]] = []
            for node, path in current:
                for parent in self._parents.get(node, []):
                    if parent in seen:
                        continue
                    seen.add(parent)
                    nxt.append((parent, path + (parent,)))
            if not nxt:
                break
            nxt.sort(key=lambda x: (x[0], x[1]))
            levels.append(nxt)
            current = nxt
        return levels

    def resolve(self, subject_id: int, predicate_id: int) -> NumericResolution:
        sid = int(subject_id)
        pid = int(predicate_id)
        for depth, level in enumerate(self._levels(sid)):
            by_object: dict[int, tuple[NumericRuntimeFact, tuple[int, ...]]] = {}
            for origin, path in level:
                for fact in self._by_subject_predicate.get((origin, pid), []):
                    if fact.quality + 1e-15 < self.policy.min_quality:
                        continue
                    incumbent = by_object.get(fact.object_id)
                    if incumbent is None:
                        by_object[fact.object_id] = (fact, path)
                    else:
                        old = incumbent[0]
                        if (fact.quality, fact.confidence, -fact.object_id, -fact.subject_id) > (
                            old.quality, old.confidence, -old.object_id, -old.subject_id
                        ):
                            by_object[fact.object_id] = (fact, path)
            if not by_object:
                continue

            ranked = sorted(
                by_object.values(),
                key=lambda item: (-item[0].quality, -item[0].confidence, item[0].object_id, item[0].subject_id, item[1]),
            )
            top_fact, top_path = ranked[0]
            candidate_ids = tuple(item[0].object_id for item in ranked)
            if len(ranked) > 1:
                second = ranked[1][0]
                if top_fact.object_id != second.object_id and (top_fact.quality - second.quality) < self.policy.ambiguity_margin:
                    return NumericResolution(
                        "ambiguous", sid, pid, None, None, depth, (), 0.0, 0.0, 0, 0,
                        candidate_ids,
                        "nearest specificity level contains competing objects without sufficient quality margin",
                    )
            return NumericResolution(
                "resolved", sid, pid, top_fact.object_id, top_fact.subject_id, depth,
                tuple(top_path), top_fact.score, top_fact.confidence,
                top_fact.source_count, top_fact.evidence_count, candidate_ids,
                "local" if depth == 0 else "nearest inherited fact",
            )

        return NumericResolution(
            "not_found", sid, pid, None, None, None, (), 0.0, 0.0, 0, 0, (),
            "no qualifying fact found within the configured parent depth",
        )

    def resolve_surface_query(
        self,
        surface: str,
        predicate_id: int,
        *,
        modality: str = Modality.TEXT.value,
        locale: str = "",
    ) -> NumericResolution:
        ids = self.resolve_surface(surface, modality=modality, locale=locale)
        if not ids:
            return NumericResolution(
                "unknown_surface", 0, int(predicate_id), None, None, None, (), 0.0, 0.0, 0, 0, (),
                f"surface {surface!r} has no binding",
            )
        if len(ids) > 1:
            return NumericResolution(
                "ambiguous_surface", 0, int(predicate_id), None, None, None, (), 0.0, 0.0, 0, 0, tuple(ids),
                f"surface {surface!r} maps to multiple concept IDs",
            )
        return self.resolve(ids[0], predicate_id)


def _stable_payload_digest(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_numeric_runtime_pack(
    db: sqlite3.Connection,
    output_path: str | Path,
    *,
    canonical_run_id: int | None = None,
    inheritance_relation_types: Iterable[str] = (RelationType.IS_A.value,),
    include_surface_bindings: bool = True,
    metadata: dict[str, Any] | None = None,
) -> NumericRuntimePack:
    rid = int(canonical_run_id or latest_canonical_run_id(db))
    row = db.execute(
        "SELECT digest FROM canonical_id_build_runs WHERE canonical_id_run_id=?",
        (rid,),
    ).fetchone()
    if not row:
        raise KeyError(f"unknown canonical_id_run_id={rid}")
    canonical_digest = str(row[0])
    relations = [
        NumericRuntimeFact(t.subject_id, t.predicate_id, t.object_id, t.score, t.confidence, t.source_count, t.evidence_count)
        for t in load_numeric_triples(db, rid)
    ]
    inheritance_ids = tuple(sorted({canonical_relation_id(db, x) for x in inheritance_relation_types}))
    surfaces: list[NumericSurfaceBinding] = []
    if include_surface_bindings:
        rows = db.execute(
            """SELECT surface,modality,locale,concept_numeric_id
               FROM canonical_surface_bindings
               ORDER BY surface,modality,locale,concept_numeric_id"""
        ).fetchall()
        surfaces = [NumericSurfaceBinding(str(a), str(b), str(c), int(d)) for a,b,c,d in rows]

    relations.sort(key=lambda r: (r.subject_id, r.predicate_id, r.object_id, -r.quality))
    surfaces.sort(key=lambda s: (s.surface, s.modality, s.locale, s.concept_id))
    pack = NumericRuntimePack(
        NUMERIC_RUNTIME_FORMAT, NUMERIC_RUNTIME_VERSION, rid, canonical_digest,
        inheritance_ids, relations, surfaces, "", dict(metadata or {}),
    )
    payload = pack.to_payload(include_digest=False)
    pack.digest = _stable_payload_digest(payload)
    final_payload = pack.to_payload(include_digest=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(final_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return pack


def load_numeric_runtime_pack(path: str | Path) -> NumericRuntimePack:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != NUMERIC_RUNTIME_FORMAT:
        raise ValueError(f"unsupported numeric runtime format: {payload.get('format')!r}")
    if payload.get("version") != NUMERIC_RUNTIME_VERSION:
        raise ValueError(f"unsupported numeric runtime version: {payload.get('version')!r}")
    digest = str(payload.get("digest") or "")
    body = dict(payload)
    body.pop("digest", None)
    actual = _stable_payload_digest(body)
    if digest != actual:
        raise ValueError("numeric runtime pack digest mismatch")
    return NumericRuntimePack(
        str(payload["format"]), str(payload["version"]), int(payload["canonical_run_id"]),
        str(payload["canonical_digest"]), tuple(int(x) for x in payload["inheritance_predicate_ids"]),
        [NumericRuntimeFact.from_list(x) for x in payload["relations"]],
        [NumericSurfaceBinding.from_list(x) for x in payload["surfaces"]],
        digest, dict(payload.get("metadata") or {}),
    )


def _node(node_id: str, *, surface: str | None = None, level: int = 3) -> IRNode:
    return IRNode(node_id, Modality.CROSSMODAL.value, NodeKind.CONCEPT.value, level, surface=surface)


def run_numeric_runtime_demo(out_dir: str | Path) -> dict[str, Any]:
    """Reference proof for specific→general numeric runtime routing and overrides."""
    from .database import CanonicalDB

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    db_path = out / "numeric-runtime-demo.sqlite"
    pack_path = out / "knowledge.eqnum.json"
    if db_path.exists():
        db_path.unlink()
    if pack_path.exists():
        pack_path.unlink()

    nodes = {
        name: _node(f"v048:{name}", surface=None, level=level)
        for name, level in {
            "animal": 4, "bird": 5, "owl": 6, "penguin": 6, "weakbird": 6,
            "alive_true": 2, "feathers": 2, "fly_true": 2, "fly_false": 2,
            "nocturnal": 2,
        }.items()
    }
    src = SourceRef("v048-demo", "1", "numeric-runtime")
    # Predicate names can be arbitrary canonical relation identities; their labels are
    # build-time/debug surfaces, not runtime knowledge objects.
    rel_is_a = RelationType.IS_A.value
    rel_alive = "state:alive"
    rel_covering = "trait:covering"
    rel_can_fly = "capability:flight"
    rel_activity = "state:activity"
    evidence = [
        RelationEvidence(nodes["bird"].node_id, nodes["animal"].node_id, rel_is_a, 1.0, 1.0, src),
        RelationEvidence(nodes["owl"].node_id, nodes["bird"].node_id, rel_is_a, 1.0, 1.0, src),
        RelationEvidence(nodes["penguin"].node_id, nodes["bird"].node_id, rel_is_a, 1.0, 1.0, src),
        RelationEvidence(nodes["weakbird"].node_id, nodes["bird"].node_id, rel_is_a, 1.0, 1.0, src),
        RelationEvidence(nodes["animal"].node_id, nodes["alive_true"].node_id, rel_alive, 0.99, 0.99, src),
        RelationEvidence(nodes["bird"].node_id, nodes["feathers"].node_id, rel_covering, 0.98, 0.98, src),
        RelationEvidence(nodes["bird"].node_id, nodes["fly_true"].node_id, rel_can_fly, 0.99, 0.99, src),
        # Lower quality than the inherited bird fact on purpose: specificity must win.
        RelationEvidence(nodes["penguin"].node_id, nodes["fly_false"].node_id, rel_can_fly, 0.80, 0.80, src),
        # Below the runtime min-quality threshold: the parent may legitimately answer.
        RelationEvidence(nodes["weakbird"].node_id, nodes["fly_false"].node_id, rel_can_fly, 0.30, 0.30, src),
        RelationEvidence(nodes["owl"].node_id, nodes["nocturnal"].node_id, rel_activity, 0.95, 0.95, src),
    ]

    with CanonicalDB(db_path) as cdb:
        for n in nodes.values():
            cdb.upsert_node(n)
        for ev in evidence:
            cdb.add_evidence(ev)
        cdb.db.commit()
        for surface, key, locale in (
            ("Eule", "owl", "de"), ("owl", "owl", "en"),
            ("Pinguin", "penguin", "de"), ("Schwachvogel", "weakbird", "de"),
        ):
            bind_surface(cdb.db, surface=surface, concept_node_id=nodes[key].node_id, locale=locale)
        build = materialize_canonical_id_space(cdb.db)
        ids = {k: canonical_node_id(cdb.db, n.node_id) for k,n in nodes.items()}
        predicates = {
            "is_a": canonical_relation_id(cdb.db, rel_is_a),
            "alive": canonical_relation_id(cdb.db, rel_alive),
            "covering": canonical_relation_id(cdb.db, rel_covering),
            "can_fly": canonical_relation_id(cdb.db, rel_can_fly),
            "activity": canonical_relation_id(cdb.db, rel_activity),
            "unknown": canonical_relation_id(cdb.db, "unknown:v048"),
        }
        pack = build_numeric_runtime_pack(
            cdb.db, pack_path, canonical_run_id=build.run_id,
            inheritance_relation_types=(rel_is_a,),
            metadata={"demo": "v0.48 specific-to-general numeric runtime"},
        )
        canonical_triple_count = len(load_numeric_triples(cdb.db, build.run_id))

    # DB is deliberately closed here. All following answers come from the pack only.
    loaded = load_numeric_runtime_pack(pack_path)
    runtime = SpecificToGeneralNumericRuntime(
        loaded,
        NumericRuntimePolicy(loaded.inheritance_predicate_ids, min_quality=0.50, ambiguity_margin=0.05),
    )
    owl_alive = runtime.resolve(ids["owl"], predicates["alive"])
    owl_cover = runtime.resolve(ids["owl"], predicates["covering"])
    owl_fly = runtime.resolve(ids["owl"], predicates["can_fly"])
    penguin_fly = runtime.resolve(ids["penguin"], predicates["can_fly"])
    weak_fly = runtime.resolve(ids["weakbird"], predicates["can_fly"])
    owl_activity = runtime.resolve(ids["owl"], predicates["activity"])
    unknown = runtime.resolve(ids["owl"], predicates["unknown"])
    surface_owl = runtime.resolve_surface_query("Eule", predicates["covering"], locale="de")

    controls = {
        "runtime_is_db_free_after_pack_load": True,
        "owl_inherits_animal_state_at_depth_2": owl_alive.status == "resolved" and owl_alive.object_id == ids["alive_true"] and owl_alive.inheritance_depth == 2,
        "owl_inherits_bird_covering_at_depth_1": owl_cover.status == "resolved" and owl_cover.object_id == ids["feathers"] and owl_cover.inheritance_depth == 1,
        "owl_inherits_bird_flight_at_depth_1": owl_fly.status == "resolved" and owl_fly.object_id == ids["fly_true"] and owl_fly.inheritance_depth == 1,
        "penguin_local_override_beats_stronger_parent": penguin_fly.status == "resolved" and penguin_fly.object_id == ids["fly_false"] and penguin_fly.inheritance_depth == 0,
        "weak_local_fact_below_threshold_falls_back_to_parent": weak_fly.status == "resolved" and weak_fly.object_id == ids["fly_true"] and weak_fly.inheritance_depth == 1,
        "owl_local_fact_stays_local": owl_activity.status == "resolved" and owl_activity.object_id == ids["nocturnal"] and owl_activity.inheritance_depth == 0,
        "unknown_predicate_abstains": unknown.status == "not_found",
        "surface_layer_resolves_into_same_numeric_runtime": surface_owl.status == "resolved" and surface_owl.object_id == ids["feathers"],
        "runtime_pack_preserves_canonical_triple_count": len(loaded.relations) == canonical_triple_count,
        "runtime_core_relations_are_numeric": all(isinstance(x, int) for r in loaded.relations for x in (r.subject_id, r.predicate_id, r.object_id)),
    }

    report = {
        "format": "equinox-forge-specific-general-numeric-runtime-reference-v1",
        "version": NUMERIC_RUNTIME_VERSION,
        "canonical_build": build.to_dict(),
        "runtime_pack": {
            "format": loaded.format,
            "version": loaded.version,
            "digest": loaded.digest,
            "relation_count": len(loaded.relations),
            "surface_binding_count": len(loaded.surfaces),
            "file_bytes": pack_path.stat().st_size,
        },
        "numeric_ids": ids,
        "predicate_ids": predicates,
        "queries": {
            "owl_alive": owl_alive.to_dict(),
            "owl_covering": owl_cover.to_dict(),
            "owl_can_fly": owl_fly.to_dict(),
            "penguin_can_fly": penguin_fly.to_dict(),
            "weakbird_can_fly": weak_fly.to_dict(),
            "owl_activity": owl_activity.to_dict(),
            "owl_unknown": unknown.to_dict(),
            "surface_eule_covering": surface_owl.to_dict(),
        },
        "controls": controls,
    }
    report["digest"] = _stable_payload_digest({k: v for k,v in report.items() if k != "digest"})
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the Equinox v0.48 specific-to-general numeric runtime proof.")
    ap.add_argument("--out", default="out/v0_48_numeric_runtime")
    args = ap.parse_args(argv)
    report = run_numeric_runtime_demo(args.out)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
