from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import argparse
import hashlib
import json
import sqlite3

from .canonical_id_space import (
    CanonicalTriple,
    canonical_relation_id,
    register_canonical_identity,
)
from .canonical_relation_facets import (
    FacetedMacroNumericRuntime,
    FacetedMacroRuntimePack,
    FamilyValue,
    RelationFacet,
    _macro_pack_from_payload,
    _insert_derived_canonical_run,
    build_faceted_macro_runtime_pack,
    materialize_relation_faceted_id_space,
)
from .macro_numeric_runtime import build_macro_numeric_runtime_pack
from .recursive_macro_collapse import RecursiveMacroPolicy, materialize_recursive_macro_collapse
from .recursive_relation_collapse import (
    RelationMacroPolicy,
    materialize_recursive_relation_macro_collapse,
)
from .structural_closure import StructuralClosurePolicy, materialize_structural_closures

RELATION_MACRO_RUNTIME_FORMAT = "equinox-relation-macro-runtime-v1"
RELATION_MACRO_RUNTIME_VERSION = "0.53"


@dataclass(frozen=True)
class RelationMacroRoute:
    macro_id: int
    level: int
    child_macro_ids: tuple[int, ...]
    direct_facet_ids: tuple[int, ...]
    descendant_facet_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "macro_id": int(self.macro_id),
            "level": int(self.level),
            "child_macro_ids": [int(x) for x in self.child_macro_ids],
            "direct_facet_ids": [int(x) for x in self.direct_facet_ids],
            "descendant_facet_ids": [int(x) for x in self.descendant_facet_ids],
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "RelationMacroRoute":
        return cls(
            int(row["macro_id"]),
            int(row["level"]),
            tuple(int(x) for x in row.get("child_macro_ids", [])),
            tuple(int(x) for x in row.get("direct_facet_ids", [])),
            tuple(int(x) for x in row.get("descendant_facet_ids", [])),
        )


@dataclass
class RelationMacroRuntimePack:
    format: str
    version: str
    faceted_pack: FacetedMacroRuntimePack
    relation_macro_run_id: int
    relation_macro_digest: str
    routes: list[RelationMacroRoute]
    digest: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            "format": self.format,
            "version": self.version,
            "faceted_pack": self.faceted_pack.to_payload(include_digest=True),
            "relation_macro_run_id": int(self.relation_macro_run_id),
            "relation_macro_digest": self.relation_macro_digest,
            "routes": [r.to_dict() for r in sorted(self.routes, key=lambda x: (x.level, x.macro_id))],
            "metadata": dict(self.metadata),
        }
        if include_digest:
            payload["digest"] = self.digest
        return payload


@dataclass(frozen=True)
class RelationMacroValue:
    facet_predicate_id: int
    object_id: int
    route_kind: str
    concept_macro_id: int | None
    origin_subject_id: int | None
    inheritance_depth: int | None
    score: float
    confidence: float
    concept_nodes_touched: int
    concept_baseline_nodes_touched: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RelationMacroResolution:
    status: str
    subject_id: int
    relation_macro_id: int
    values: tuple[RelationMacroValue, ...]
    candidate_facets_examined: int
    unindexed_relation_space_candidates: int
    route_macro_nodes: int
    route_total_nodes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "subject_id": int(self.subject_id),
            "relation_macro_id": int(self.relation_macro_id),
            "values": [v.to_dict() for v in self.values],
            "candidate_facets_examined": int(self.candidate_facets_examined),
            "unindexed_relation_space_candidates": int(self.unindexed_relation_space_candidates),
            "route_macro_nodes": int(self.route_macro_nodes),
            "route_total_nodes": int(self.route_total_nodes),
        }


def _stable_digest(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _faceted_pack_from_payload(payload: dict[str, Any]) -> FacetedMacroRuntimePack:
    digest = str(payload.get("digest") or "")
    body = dict(payload)
    body.pop("digest", None)
    if digest != _stable_digest(body):
        raise ValueError("embedded faceted runtime pack digest mismatch")
    return FacetedMacroRuntimePack(
        str(payload["format"]),
        str(payload["version"]),
        _macro_pack_from_payload(dict(payload["macro_pack"])),
        [RelationFacet.from_list(x) for x in payload["facets"]],
        digest,
        dict(payload.get("metadata") or {}),
    )


def _descendant_facets(
    macro_id: int,
    direct_macros: dict[int, tuple[int, ...]],
    direct_facets: dict[int, tuple[int, ...]],
    *,
    _seen: set[int] | None = None,
) -> tuple[int, ...]:
    seen = set(_seen or set())
    mid = int(macro_id)
    if mid in seen:
        return ()
    seen.add(mid)
    out = set(int(x) for x in direct_facets.get(mid, ()))
    for child in direct_macros.get(mid, ()):
        out.update(_descendant_facets(child, direct_macros, direct_facets, _seen=seen))
    return tuple(sorted(out))


def build_relation_macro_runtime_pack(
    db: sqlite3.Connection,
    output_path: str | Path,
    *,
    faceted_pack: FacetedMacroRuntimePack,
    relation_macro_run_id: int,
    metadata: dict[str, Any] | None = None,
) -> RelationMacroRuntimePack:
    rid = int(relation_macro_run_id)
    row = db.execute(
        "SELECT digest FROM recursive_relation_macro_runs WHERE relation_macro_run_id=?",
        (rid,),
    ).fetchone()
    if not row:
        raise KeyError(f"unknown relation_macro_run_id={rid}")
    relation_macro_digest = str(row[0])

    macro_rows = db.execute(
        """SELECT macro_id,level FROM recursive_relation_macros
           WHERE relation_macro_run_id=? ORDER BY level,macro_id""",
        (rid,),
    ).fetchall()
    direct_macros: dict[int, list[int]] = {}
    direct_facets: dict[int, list[int]] = {}
    for macro_id, member_id, member_kind in db.execute(
        """SELECT macro_id,member_id,member_kind FROM recursive_relation_macro_members
           WHERE relation_macro_run_id=? ORDER BY macro_id,position,member_id""",
        (rid,),
    ).fetchall():
        target = direct_macros if str(member_kind) == "relation_macro" else direct_facets
        target.setdefault(int(macro_id), []).append(int(member_id))
    dm = {k: tuple(sorted(set(v))) for k, v in direct_macros.items()}
    df = {k: tuple(sorted(set(v))) for k, v in direct_facets.items()}

    routes: list[RelationMacroRoute] = []
    valid_facets = {int(f.facet_predicate_id) for f in faceted_pack.facets}
    for macro_id, level in macro_rows:
        descendants = _descendant_facets(int(macro_id), dm, df)
        if not set(descendants).issubset(valid_facets):
            raise ValueError(f"relation macro {macro_id} references facet outside faceted runtime pack")
        routes.append(RelationMacroRoute(
            int(macro_id), int(level), dm.get(int(macro_id), ()), df.get(int(macro_id), ()), descendants,
        ))

    pack = RelationMacroRuntimePack(
        RELATION_MACRO_RUNTIME_FORMAT,
        RELATION_MACRO_RUNTIME_VERSION,
        faceted_pack,
        rid,
        relation_macro_digest,
        routes,
        "",
        dict(metadata or {}),
    )
    pack.digest = _stable_digest(pack.to_payload(include_digest=False))
    Path(output_path).write_text(
        json.dumps(pack.to_payload(include_digest=True), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return pack


def load_relation_macro_runtime_pack(path: str | Path) -> RelationMacroRuntimePack:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != RELATION_MACRO_RUNTIME_FORMAT or payload.get("version") != RELATION_MACRO_RUNTIME_VERSION:
        raise ValueError("unsupported relation macro runtime pack")
    digest = str(payload.get("digest") or "")
    body = dict(payload)
    body.pop("digest", None)
    if digest != _stable_digest(body):
        raise ValueError("relation macro runtime pack digest mismatch")
    return RelationMacroRuntimePack(
        str(payload["format"]),
        str(payload["version"]),
        _faceted_pack_from_payload(dict(payload["faceted_pack"])),
        int(payload["relation_macro_run_id"]),
        str(payload["relation_macro_digest"]),
        [RelationMacroRoute.from_dict(x) for x in payload["routes"]],
        digest,
        dict(payload.get("metadata") or {}),
    )


class RelationMacroNumericRuntime:
    """Execute recursive relation-macro IDs as numeric predicate-set routers.

    This layer does not grant relation macros new truth authority. A macro only
    selects descendant facet predicates; each concrete value is still resolved by
    the frozen v0.51/v0.49 concept-side runtime. Thus relation macros are routing
    handles over predicates, not an additional evidence source.
    """

    def __init__(self, pack: RelationMacroRuntimePack):
        self.pack = pack
        self.faceted_runtime = FacetedMacroNumericRuntime(pack.faceted_pack)
        self._routes = {int(r.macro_id): r for r in pack.routes}
        self._all_facet_ids = tuple(sorted(int(f.facet_predicate_id) for f in pack.faceted_pack.facets))
        self._facet_family = {int(f.facet_predicate_id): int(f.family_predicate_id) for f in pack.faceted_pack.facets}

    def resolve_relation_macro(self, subject_id: int, relation_macro_id: int) -> RelationMacroResolution:
        sid, mid = int(subject_id), int(relation_macro_id)
        route = self._routes.get(mid)
        if route is None:
            return RelationMacroResolution("unknown_relation_macro", sid, mid, (), 0, len(self._all_facet_ids), 0, 0)

        values: list[RelationMacroValue] = []
        for facet_id in route.descendant_facet_ids:
            result = self.faceted_runtime.resolve(sid, int(facet_id))
            r = result.resolution
            if r.status != "resolved" or r.object_id is None:
                continue
            values.append(RelationMacroValue(
                int(facet_id), int(r.object_id), str(result.route_kind), result.macro_id,
                r.origin_subject_id, r.inheritance_depth, float(r.score), float(r.confidence),
                int(result.nodes_touched), int(result.baseline_nodes_touched),
            ))
        values.sort(key=lambda x: (
            10**9 if x.inheritance_depth is None else int(x.inheritance_depth),
            int(x.facet_predicate_id), int(x.object_id),
        ))
        # The pack contains the transitive leaf set, so runtime routing is one macro
        # index lookup plus exactly the relevant facet candidates. The recursive
        # macro tree remains in the pack for audit/explanation.
        return RelationMacroResolution(
            "resolved" if values else "not_found",
            sid,
            mid,
            tuple(values),
            len(route.descendant_facet_ids),
            len(self._all_facet_ids),
            1,
            1 + len(route.descendant_facet_ids),
        )

    def resolve_surface_relation_macro(self, surface: str, relation_macro_id: int, *, modality: str = "text", locale: str = "") -> RelationMacroResolution:
        ids = self.faceted_runtime.resolve_surface(surface, modality=modality, locale=locale)
        if len(ids) != 1:
            return RelationMacroResolution("unknown_or_ambiguous_surface", 0, int(relation_macro_id), (), 0, len(self._all_facet_ids), 0, 0)
        return self.resolve_relation_macro(ids[0], relation_macro_id)


def _build_demo(out: Path) -> tuple[dict[str, Any], RelationMacroRuntimePack]:
    db_path = out / "forge.sqlite"
    faceted_path = out / "faceted.eqnum.json"
    runtime_path = out / "relation-macro.eqnum.json"
    if db_path.exists():
        db_path.unlink()
    db = sqlite3.connect(db_path)

    is_a = canonical_relation_id(db, "is_a")
    prop = canonical_relation_id(db, "property")
    action = canonical_relation_id(db, "action")
    names = [
        "owl", "falcon", "dog", "wolf", "bird", "mammal", "animal", "living",
        "feathers", "fur", "body", "cells", "fly", "run", "move", "live",
    ]
    ids = {n: register_canonical_identity(db, namespace="node", symbolic_id=n, identity_kind="concept") for n in names}
    triples = [
        CanonicalTriple(ids["owl"], is_a, ids["bird"], 1, 1, 1, 1),
        CanonicalTriple(ids["falcon"], is_a, ids["bird"], 1, 1, 1, 1),
        CanonicalTriple(ids["dog"], is_a, ids["mammal"], 1, 1, 1, 1),
        CanonicalTriple(ids["wolf"], is_a, ids["mammal"], 1, 1, 1, 1),
        CanonicalTriple(ids["bird"], is_a, ids["animal"], 1, 1, 1, 1),
        CanonicalTriple(ids["mammal"], is_a, ids["animal"], 1, 1, 1, 1),
        CanonicalTriple(ids["animal"], is_a, ids["living"], 1, 1, 1, 1),
        CanonicalTriple(ids["bird"], prop, ids["feathers"], .99, .99, 1, 1),
        CanonicalTriple(ids["mammal"], prop, ids["fur"], .99, .99, 1, 1),
        CanonicalTriple(ids["animal"], prop, ids["body"], .98, .98, 1, 1),
        CanonicalTriple(ids["living"], prop, ids["cells"], .97, .97, 1, 1),
        CanonicalTriple(ids["bird"], action, ids["fly"], .99, .99, 1, 1),
        CanonicalTriple(ids["mammal"], action, ids["run"], .99, .99, 1, 1),
        CanonicalTriple(ids["animal"], action, ids["move"], .98, .98, 1, 1),
        CanonicalTriple(ids["living"], action, ids["live"], .97, .97, 1, 1),
    ]
    source_run, _ = _insert_derived_canonical_run(db, source_run_id=0, triples=triples, metadata={"fixture": "v053"})
    facets = materialize_relation_faceted_id_space(db, source_canonical_run_id=source_run)
    roots = [ids[x] for x in ("owl", "falcon", "dog", "wolf")]
    closure = materialize_structural_closures(
        db, canonical_run_id=facets.derived_canonical_run_id, root_ids=roots,
        policy=StructuralClosurePolicy(max_parent_depth=8),
    )
    concept_macro = materialize_recursive_macro_collapse(
        db, canonical_run_id=facets.derived_canonical_run_id, structural_closure_run_id=closure.run_id,
        root_ids=roots, policy=RecursiveMacroPolicy(min_members=2, min_common_facts=2, max_levels=4),
    )
    macro_pack = build_macro_numeric_runtime_pack(
        db, out / "concept-macro.eqnum.json", canonical_run_id=facets.derived_canonical_run_id,
        recursive_macro_run_id=concept_macro.run_id,
    )
    faceted_pack = build_faceted_macro_runtime_pack(
        db, faceted_path, macro_pack=macro_pack, relation_facet_run_id=facets.run_id,
    )
    relation_macro = materialize_recursive_relation_macro_collapse(
        db, relation_facet_run_id=facets.run_id,
        policy=RelationMacroPolicy(min_members=2, min_common_meta_facts=1, max_levels=4),
    )
    runtime_pack = build_relation_macro_runtime_pack(
        db, runtime_path, faceted_pack=faceted_pack, relation_macro_run_id=relation_macro.run_id,
        metadata={"trial": RELATION_MACRO_RUNTIME_VERSION},
    )

    relation_is_a = canonical_relation_id(db, "relation_is_a")
    macro_facts = {
        int(mid): set((int(p), int(o)) for p, o in db.execute(
            "SELECT predicate_id,object_id FROM recursive_relation_macro_meta WHERE relation_macro_run_id=? AND macro_id=?",
            (relation_macro.run_id, int(mid)),
        ).fetchall())
        for mid, in db.execute(
            "SELECT macro_id FROM recursive_relation_macros WHERE relation_macro_run_id=? ORDER BY macro_id",
            (relation_macro.run_id,),
        ).fetchall()
    }
    # Lower macros retain family-specific IS_A residuals after the level-2 collapse.
    property_macro = next(mid for mid, facts in macro_facts.items() if (relation_is_a, prop) in facts)
    action_macro = next(mid for mid, facts in macro_facts.items() if (relation_is_a, action) in facts)
    levels = {int(mid): int(level) for mid, level in db.execute(
        "SELECT macro_id,level FROM recursive_relation_macros WHERE relation_macro_run_id=?",
        (relation_macro.run_id,),
    ).fetchall()}
    root_macro = next(mid for mid, level in levels.items() if level == max(levels.values()))

    report = {
        "format": RELATION_MACRO_RUNTIME_FORMAT,
        "version": RELATION_MACRO_RUNTIME_VERSION,
        "facet_count": len(facets and runtime_pack.faceted_pack.facets),
        "relation_macro_count": relation_macro.macro_count,
        "relation_macro_levels": relation_macro.levels_built,
        "property_macro_id": property_macro,
        "action_macro_id": action_macro,
        "root_relation_macro_id": root_macro,
        "pack_digest": runtime_pack.digest,
    }
    db.commit()
    db.close()
    return report, runtime_pack


def run_relation_macro_runtime_demo(out_dir: str | Path) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary, pack = _build_demo(out)
    runtime = RelationMacroNumericRuntime(load_relation_macro_runtime_pack(out / "relation-macro.eqnum.json"))

    owl_ids = runtime.faceted_runtime.resolve_surface("owl")
    # Demo canonical runs have no surface bindings, so use the deterministic ID from the embedded base by looking it up in metadata-less graph.
    # The subject id is recoverable from the hierarchy because owl is the only leaf used below; for the CLI report we persist it in metadata.
    # Re-open the pack's base relations to infer a leaf with maximum inherited family values deterministically.
    subjects = sorted({int(r.subject_id) for r in pack.faceted_pack.macro_pack.base.relations})
    # Pick the leaf that resolves to feathers/fly; this is deterministic and avoids string dependence in runtime semantics.
    prop_mid = int(summary["property_macro_id"])
    act_mid = int(summary["action_macro_id"])
    root_mid = int(summary["root_relation_macro_id"])
    chosen = None
    for sid in subjects:
        p = runtime.resolve_relation_macro(sid, prop_mid)
        a = runtime.resolve_relation_macro(sid, act_mid)
        if len(p.values) >= 3 and len(a.values) >= 3:
            chosen = sid
            break
    if chosen is None:
        raise AssertionError("could not find demo leaf subject")

    prop_result = runtime.resolve_relation_macro(chosen, prop_mid)
    act_result = runtime.resolve_relation_macro(chosen, act_mid)
    root_result = runtime.resolve_relation_macro(chosen, root_mid)
    # Canonical relation IDs are deterministic across DBs; recover family IDs without Forge state.
    tmp = sqlite3.connect(":memory:")
    try:
        property_family_id = canonical_relation_id(tmp, "property")
        action_family_id = canonical_relation_id(tmp, "action")
    finally:
        tmp.close()
    prop_family = runtime.faceted_runtime.resolve_family(chosen, property_family_id)
    act_family = runtime.faceted_runtime.resolve_family(chosen, action_family_id)

    prop_objects = {v.object_id for v in prop_result.values}
    act_objects = {v.object_id for v in act_result.values}
    root_objects = {v.object_id for v in root_result.values}
    family_prop_objects = {v.object_id for v in prop_family.values}
    family_act_objects = {v.object_id for v in act_family.values}

    total_facets = len(pack.faceted_pack.facets)
    candidate_reduction = 0.0 if total_facets == 0 else 1.0 - (prop_result.candidate_facets_examined / total_facets)
    family_index_candidates = len([f for f in pack.faceted_pack.facets if int(f.family_predicate_id) == property_family_id])
    report = {
        **summary,
        "subject_id": chosen,
        "property_query": prop_result.to_dict(),
        "action_query": act_result.to_dict(),
        "root_query": root_result.to_dict(),
        "work": {
            "unindexed_relation_space_candidates": total_facets,
            "property_macro_candidates": prop_result.candidate_facets_examined,
            "property_family_index_candidates": family_index_candidates,
            "candidate_reduction_vs_unindexed_scan": candidate_reduction,
            "macro_beats_specialized_family_index": prop_result.candidate_facets_examined < family_index_candidates,
            "macro_matches_specialized_family_index_candidate_count": prop_result.candidate_facets_examined == family_index_candidates,
        },
        "controls": {
            "property_macro_matches_family_semantics": prop_objects == family_prop_objects and len(prop_objects) == len(prop_result.values),
            "action_macro_matches_family_semantics": act_objects == family_act_objects and len(act_objects) == len(act_result.values),
            "root_macro_is_union_of_child_relation_macros": root_objects == (prop_objects | act_objects),
            "relation_macro_is_general_numeric_query_handle": prop_result.status == "resolved" and act_result.status == "resolved",
            "candidate_space_reduced_vs_unindexed_relation_scan": candidate_reduction > 0.0,
            "no_false_speedup_claim_vs_existing_family_index": prop_result.candidate_facets_examined == family_index_candidates,
        },
    }
    if not all(report["controls"].values()):
        raise AssertionError(report["controls"])
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Equinox v0.53 relation-macro numeric runtime routing demo.")
    ap.add_argument("--out", default="out/v0_53_relation_macro_runtime")
    args = ap.parse_args(argv)
    report = run_relation_macro_runtime_demo(args.out)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
