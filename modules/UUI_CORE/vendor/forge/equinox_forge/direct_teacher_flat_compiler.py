from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import argparse
import gc
import hashlib
import json
import shutil
import sqlite3
import tempfile

from .canonical_id_space import (
    canonical_node_id,
    canonical_relation_id,
    load_numeric_triples,
    materialize_canonical_id_space,
)
from .canonical_relation_facets import (
    load_relation_facets,
    materialize_relation_faceted_id_space,
    build_faceted_macro_runtime_pack,
)
from .database import CanonicalDB
from .ir import Modality, RelationType
from .macro_numeric_runtime import (
    _best_macro_object,
    build_macro_numeric_runtime_pack,
)
from .numeric_runtime import (
    NumericRuntimeFact,
    NumericRuntimePack,
    NumericSurfaceBinding,
    SpecificToGeneralNumericRuntime,
    _stable_payload_digest,
)
from .real_teacher_domain_trial import _NeuralDomainTeacher
from .recursive_macro_collapse import (
    RecursiveMacroPolicy,
    effective_macro_facts,
    materialize_recursive_macro_collapse,
)
from .recursive_relation_collapse import (
    RelationMacroPolicy,
    materialize_recursive_relation_macro_collapse,
)
from .relation_macro_runtime import build_relation_macro_runtime_pack
from .structural_closure import StructuralClosurePolicy, materialize_structural_closures
from .teacher_numeric_compiler import _load_fixture, _policy
from .top_down_teacher_structure import _concept_node_id, extract_top_down_teacher_structure
from .unified_numeric_kernel import (
    KernelFact,
    KernelRoute,
    KernelShortcut,
    KernelSurfaceBinding,
    UnifiedNumericExecutionKernel,
    UnifiedNumericKernelPack,
    UNIFIED_NUMERIC_KERNEL_FORMAT,
    UNIFIED_NUMERIC_KERNEL_VERSION,
    _stable_digest as _kernel_digest,
    build_unified_numeric_kernel_pack,
    load_unified_numeric_kernel_pack,
)
from .unified_relation_runtime import build_unified_relation_runtime_pack

DIRECT_TEACHER_FLAT_COMPILER_FORMAT = "equinox-direct-teacher-flat-kernel-compiler-v1"
DIRECT_TEACHER_FLAT_COMPILER_VERSION = "0.56"


@dataclass(frozen=True)
class DirectFlatCompileSummary:
    canonical_run_id: int
    concept_macro_run_id: int
    relation_facet_run_id: int
    relation_macro_run_id: int
    fact_count: int
    surface_count: int
    concept_shortcut_count: int
    route_count: int
    relation_macro_route_count: int
    family_alias_route_count: int
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_digest(db: sqlite3.Connection, run_id: int) -> str:
    row = db.execute(
        "SELECT digest FROM canonical_id_build_runs WHERE canonical_id_run_id=?",
        (int(run_id),),
    ).fetchone()
    if not row:
        raise KeyError(f"unknown canonical_id_run_id={run_id}")
    return str(row[0])


def _numeric_surfaces(db: sqlite3.Connection) -> list[NumericSurfaceBinding]:
    rows = db.execute(
        """SELECT surface,modality,locale,concept_numeric_id
           FROM canonical_surface_bindings
           ORDER BY surface,modality,locale,concept_numeric_id"""
    ).fetchall()
    return [NumericSurfaceBinding(str(a), str(b), str(c), int(d)) for a, b, c, d in rows]


def _direct_base_pack(db: sqlite3.Connection, canonical_run_id: int) -> NumericRuntimePack:
    rid = int(canonical_run_id)
    relations = [
        NumericRuntimeFact(
            int(t.subject_id), int(t.predicate_id), int(t.object_id),
            float(t.score), float(t.confidence), int(t.source_count), int(t.evidence_count),
        )
        for t in load_numeric_triples(db, rid)
    ]
    relations.sort(key=lambda r: (r.subject_id, r.predicate_id, r.object_id, -r.quality))
    surfaces = _numeric_surfaces(db)
    surfaces.sort(key=lambda s: (s.surface, s.modality, s.locale, s.concept_id))
    inheritance = (int(canonical_relation_id(db, RelationType.IS_A.value)),)
    pack = NumericRuntimePack(
        "equinox-specific-general-numeric-runtime-v1",
        "0.48",
        rid,
        _canonical_digest(db, rid),
        inheritance,
        relations,
        surfaces,
        "",
        {"compiler_only": DIRECT_TEACHER_FLAT_COMPILER_VERSION},
    )
    pack.digest = _stable_payload_digest(pack.to_payload(include_digest=False))
    return pack


def _compile_concept_shortcuts(
    db: sqlite3.Connection,
    *,
    base: NumericRuntimePack,
    concept_macro_run_id: int,
    min_quality: float,
    ambiguity_margin: float,
) -> list[KernelShortcut]:
    runtime = SpecificToGeneralNumericRuntime(base)
    # Keep policy values aligned with the emitted flat kernel.
    runtime.policy = type(runtime.policy)(
        tuple(base.inheritance_predicate_ids),
        runtime.policy.max_parent_depth,
        float(min_quality),
        float(ambiguity_margin),
    )
    concept_routes = [
        (int(member), int(macro))
        for member, macro in db.execute(
            """SELECT member_id,macro_id FROM recursive_macro_members
               WHERE recursive_macro_run_id=? AND member_kind='concept'
               ORDER BY member_id,macro_id""",
            (int(concept_macro_run_id),),
        ).fetchall()
    ]
    predicates = sorted({
        int(r.predicate_id)
        for r in base.relations
        if int(r.predicate_id) not in set(base.inheritance_predicate_ids)
    })
    cache: dict[int, list[Any]] = {}
    out: list[KernelShortcut] = []
    for concept_id, macro_id in concept_routes:
        macro_facts = cache.setdefault(
            macro_id,
            list(effective_macro_facts(
                db, macro_id, recursive_macro_run_id=int(concept_macro_run_id)
            ).values()),
        )
        for pid in predicates:
            baseline = runtime.resolve(concept_id, pid)
            if baseline.status != "resolved" or baseline.object_id is None or not baseline.inherited:
                continue
            candidate = _best_macro_object(
                macro_facts,
                pid,
                min_quality=float(min_quality),
                ambiguity_margin=float(ambiguity_margin),
            )
            if candidate is None or int(candidate[0]) != int(baseline.object_id):
                continue
            out.append(KernelShortcut(
                int(concept_id), int(pid), int(macro_id), int(baseline.object_id),
                int(baseline.origin_subject_id or concept_id), int(baseline.inheritance_depth or 0),
                tuple(int(x) for x in baseline.inheritance_path),
                float(baseline.score), float(baseline.confidence),
                int(baseline.source_count), int(baseline.evidence_count),
            ))
    out.sort(key=lambda s: (s.concept_id, s.predicate_id, s.macro_id, s.object_id))
    return out


def _relation_macro_descendants(
    db: sqlite3.Connection,
    *,
    relation_macro_run_id: int,
) -> tuple[dict[int, tuple[int, ...]], dict[int, int]]:
    rid = int(relation_macro_run_id)
    macro_levels = {
        int(mid): int(level)
        for mid, level in db.execute(
            """SELECT macro_id,level FROM recursive_relation_macros
               WHERE relation_macro_run_id=? ORDER BY level,macro_id""",
            (rid,),
        ).fetchall()
    }
    direct_macros: dict[int, list[int]] = {}
    direct_facets: dict[int, list[int]] = {}
    for macro_id, member_id, member_kind in db.execute(
        """SELECT macro_id,member_id,member_kind FROM recursive_relation_macro_members
           WHERE relation_macro_run_id=? ORDER BY macro_id,position,member_id""",
        (rid,),
    ).fetchall():
        target = direct_macros if str(member_kind) == "relation_macro" else direct_facets
        target.setdefault(int(macro_id), []).append(int(member_id))

    memo: dict[int, tuple[int, ...]] = {}

    def descendants(mid: int, seen: set[int] | None = None) -> tuple[int, ...]:
        if mid in memo:
            return memo[mid]
        trail = set(seen or set())
        if mid in trail:
            return ()
        trail.add(mid)
        values = set(int(x) for x in direct_facets.get(mid, []))
        for child in direct_macros.get(mid, []):
            values.update(descendants(int(child), trail))
        result = tuple(sorted(values))
        memo[mid] = result
        return result

    for mid in sorted(macro_levels):
        descendants(mid)
    return memo, macro_levels


def _compile_relation_routes(
    db: sqlite3.Connection,
    *,
    base: NumericRuntimePack,
    relation_facet_run_id: int,
    relation_macro_run_id: int,
) -> tuple[list[KernelRoute], int, int]:
    facets = load_relation_facets(db, int(relation_facet_run_id))
    facet_ids = {int(f.facet_predicate_id) for f in facets}
    scalar_ids = {int(r.predicate_id) for r in base.relations} | facet_ids
    routes: dict[int, KernelRoute] = {
        rid: KernelRoute(rid, "scalar", (rid,)) for rid in sorted(scalar_ids)
    }

    descendants, levels = _relation_macro_descendants(
        db, relation_macro_run_id=int(relation_macro_run_id)
    )
    macro_route_count = 0
    for mid in sorted(descendants):
        concrete = descendants[mid]
        if not concrete:
            continue
        routes[mid] = KernelRoute(mid, "relation_set", concrete)
        macro_route_count += 1

    facets_by_family: dict[int, set[int]] = {}
    for facet in facets:
        facets_by_family.setdefault(int(facet.family_predicate_id), set()).add(int(facet.facet_predicate_id))
    macros_by_leafset: dict[frozenset[int], list[tuple[int, int]]] = {}
    for mid, concrete in descendants.items():
        if concrete:
            macros_by_leafset.setdefault(frozenset(concrete), []).append((int(levels.get(mid, 10**9)), int(mid)))

    alias_count = 0
    for family_id, members in sorted(facets_by_family.items()):
        matches = sorted(macros_by_leafset.get(frozenset(members), []))
        if not matches:
            continue
        best_level = matches[0][0]
        best = [mid for level, mid in matches if level == best_level]
        if len(best) != 1:
            continue
        mid = int(best[0])
        routes[int(family_id)] = KernelRoute(int(family_id), "relation_set", descendants[mid])
        alias_count += 1

    return [routes[k] for k in sorted(routes)], macro_route_count, alias_count


def compile_flat_kernel_direct(
    db: sqlite3.Connection,
    output_path: str | Path,
    *,
    canonical_run_id: int,
    concept_macro_run_id: int,
    relation_facet_run_id: int,
    relation_macro_run_id: int,
    min_quality: float = 0.50,
    ambiguity_margin: float = 0.05,
    max_parent_depth: int = 8,
    metadata: dict[str, Any] | None = None,
) -> tuple[UnifiedNumericKernelPack, DirectFlatCompileSummary]:
    """Compile Forge build-state directly into the flat v0.55 kernel format.

    No v0.48-v0.54 runtime pack is created or required. Those historical pack
    builders remain useful as independent regression oracles only.
    """
    base = _direct_base_pack(db, int(canonical_run_id))
    shortcuts = _compile_concept_shortcuts(
        db,
        base=base,
        concept_macro_run_id=int(concept_macro_run_id),
        min_quality=float(min_quality),
        ambiguity_margin=float(ambiguity_margin),
    )
    routes, relation_macro_routes, family_aliases = _compile_relation_routes(
        db,
        base=base,
        relation_facet_run_id=int(relation_facet_run_id),
        relation_macro_run_id=int(relation_macro_run_id),
    )
    pack = UnifiedNumericKernelPack(
        UNIFIED_NUMERIC_KERNEL_FORMAT,
        UNIFIED_NUMERIC_KERNEL_VERSION,  # execution format remains v0.55
        tuple(int(x) for x in base.inheritance_predicate_ids),
        [KernelFact(*r.to_list()) for r in base.relations],
        [KernelSurfaceBinding(*s.to_list()) for s in base.surfaces],
        shortcuts,
        routes,
        float(min_quality),
        float(ambiguity_margin),
        int(max_parent_depth),
        "",
        {
            "direct_compiler_version": DIRECT_TEACHER_FLAT_COMPILER_VERSION,
            "canonical_run_id": int(canonical_run_id),
            "concept_macro_run_id": int(concept_macro_run_id),
            "relation_facet_run_id": int(relation_facet_run_id),
            "relation_macro_run_id": int(relation_macro_run_id),
            **dict(metadata or {}),
        },
    )
    pack.digest = _kernel_digest(pack.to_payload(include_digest=False))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(pack.to_payload(include_digest=True), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    summary = DirectFlatCompileSummary(
        int(canonical_run_id), int(concept_macro_run_id), int(relation_facet_run_id), int(relation_macro_run_id),
        len(pack.facts), len(pack.surfaces), len(pack.concept_shortcuts), len(pack.routes),
        int(relation_macro_routes), int(family_aliases), pack.digest,
    )
    return pack, summary


def _core_signature(pack: UnifiedNumericKernelPack) -> dict[str, Any]:
    return {
        "inheritance": list(pack.inheritance_predicate_ids),
        "facts": [x.to_list() for x in pack.facts],
        "surfaces": [x.to_list() for x in pack.surfaces],
        "shortcuts": [x.to_list() for x in pack.concept_shortcuts],
        "routes": [x.to_list() for x in sorted(pack.routes, key=lambda r: r.query_relation_id)],
        "policy": [pack.min_quality, pack.ambiguity_margin, pack.max_parent_depth],
    }


def _build_legacy_reference(
    db: sqlite3.Connection,
    *,
    canonical_run_id: int,
    concept_macro_run_id: int,
    relation_facet_run_id: int,
    relation_macro_run_id: int,
) -> UnifiedNumericKernelPack:
    """Regression oracle only; all temporary nested artifacts are deleted on exit."""
    with tempfile.TemporaryDirectory(prefix="eq-v056-oracle-") as td:
        root = Path(td)
        macro = build_macro_numeric_runtime_pack(
            db, root / "macro.json", canonical_run_id=canonical_run_id,
            recursive_macro_run_id=concept_macro_run_id,
        )
        faceted = build_faceted_macro_runtime_pack(
            db, root / "faceted.json", macro_pack=macro,
            relation_facet_run_id=relation_facet_run_id,
        )
        relmacro = build_relation_macro_runtime_pack(
            db, root / "relation-macro.json", faceted_pack=faceted,
            relation_macro_run_id=relation_macro_run_id,
        )
        unified = build_unified_relation_runtime_pack(
            root / "unified.json", relation_macro_pack=relmacro,
        )
        return build_unified_numeric_kernel_pack(
            root / "kernel.json", unified_pack=unified,
        )


def _expected_family_values(fixture: dict[str, Any]) -> dict[str, dict[str, set[str]]]:
    pairs = {str(a): str(b) for a, b in fixture["training_pairs"]}
    parent = {str(a): str(b) for a, b in fixture["expected_parent_edges"]}
    out: dict[str, dict[str, set[str]]] = {}
    for leaf in sorted(str(x) for x in fixture["seed_surfaces"]):
        cur = leaf
        seen: set[str] = set()
        props: set[str] = set()
        actions: set[str] = set()
        while cur in parent and cur not in seen:
            seen.add(cur)
            cur = parent[cur]
            p = pairs.get(f"{cur} hat")
            a = pairs.get(f"{cur} kann")
            if p:
                props.add(p)
            if a:
                actions.add(a)
        out[leaf] = {RelationType.PROPERTY.value: props, RelationType.ACTION.value: actions}
    return out


def run_direct_teacher_flat_compiler_trial(
    out_dir: str | Path,
    *,
    fixture_path: str | Path | None = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fixture = _load_fixture(fixture_path)
    teacher = _NeuralDomainTeacher(
        [(str(a), str(b)) for a, b in fixture["training_pairs"]],
        out / "teacher",
        seed=5601,
        max_epochs=800,
    )
    teacher_manifest = teacher.manifest()
    teacher_state = Path(teacher_manifest["model_dir"])
    teacher_state_bytes = teacher_state.stat().st_size
    policy = _policy(fixture)

    db_path = out / "forge.sqlite"
    final_path = out / "knowledge.eqkernel.json"
    for path in (db_path, final_path):
        if path.exists():
            path.unlink()

    with CanonicalDB(db_path) as cdb:
        structure = extract_top_down_teacher_structure(
            cdb.db,
            teacher,
            seed_surfaces=fixture["seed_surfaces"],
            source_id="v056-neural-teacher",
            source_version="1",
            policy=policy,
            metadata={"trial": DIRECT_TEACHER_FLAT_COMPILER_FORMAT},
        )
        base = materialize_canonical_id_space(cdb.db)
        facets = materialize_relation_faceted_id_space(
            cdb.db,
            source_canonical_run_id=base.run_id,
            metadata={"trial": DIRECT_TEACHER_FLAT_COMPILER_FORMAT},
        )
        seed_numeric = {
            str(surface): canonical_node_id(
                cdb.db, _concept_node_id(policy.extraction_namespace, str(surface))
            )
            for surface in fixture["seed_surfaces"]
        }
        closure = materialize_structural_closures(
            cdb.db,
            canonical_run_id=facets.derived_canonical_run_id,
            root_ids=seed_numeric.values(),
            policy=StructuralClosurePolicy(max_parent_depth=8),
        )
        concept_macro = materialize_recursive_macro_collapse(
            cdb.db,
            canonical_run_id=facets.derived_canonical_run_id,
            structural_closure_run_id=closure.run_id,
            root_ids=seed_numeric.values(),
            policy=RecursiveMacroPolicy(min_members=2, min_common_facts=2, max_levels=4),
        )
        relation_macro = materialize_recursive_relation_macro_collapse(
            cdb.db,
            relation_facet_run_id=facets.run_id,
            policy=RelationMacroPolicy(min_members=2, min_common_meta_facts=1, max_levels=4),
            metadata={"trial": DIRECT_TEACHER_FLAT_COMPILER_FORMAT},
        )
        direct_pack, direct_summary = compile_flat_kernel_direct(
            cdb.db,
            final_path,
            canonical_run_id=facets.derived_canonical_run_id,
            concept_macro_run_id=concept_macro.run_id,
            relation_facet_run_id=facets.run_id,
            relation_macro_run_id=relation_macro.run_id,
            metadata={"trial": DIRECT_TEACHER_FLAT_COMPILER_VERSION},
        )
        oracle = _build_legacy_reference(
            cdb.db,
            canonical_run_id=facets.derived_canonical_run_id,
            concept_macro_run_id=concept_macro.run_id,
            relation_facet_run_id=facets.run_id,
            relation_macro_run_id=relation_macro.run_id,
        )
        parity = _core_signature(direct_pack) == _core_signature(oracle)

    # Runtime must survive after all teacher/Forge state disappears.
    if teacher_state.exists():
        teacher_state.unlink()
    del teacher
    gc.collect()
    try:
        db_path.unlink()
    except FileNotFoundError:
        pass

    loaded = load_unified_numeric_kernel_pack(final_path)
    runtime = UnifiedNumericExecutionKernel(loaded)
    expected = _expected_family_values(fixture)

    # Only canonical IDs are used by runtime. Strings below are test-side I/O labels.
    # Use a detached in-memory ID registry for test-side symbolic lookup.
    tmp = sqlite3.connect(":memory:")
    try:
        property_id = canonical_relation_id(tmp, RelationType.PROPERTY.value)
        action_id = canonical_relation_id(tmp, RelationType.ACTION.value)
        is_a_id = canonical_relation_id(tmp, RelationType.IS_A.value)
    finally:
        tmp.close()

    query_rows: list[dict[str, Any]] = []
    all_correct = True
    for surface in sorted(expected):
        sids = runtime.resolve_surface_ids(surface, modality=Modality.TEXT.value, locale="")
        if len(sids) != 1:
            raise AssertionError((surface, sids))
        for family_name, family_id in ((RelationType.PROPERTY.value, property_id), (RelationType.ACTION.value, action_id)):
            result = runtime.resolve(sids[0], family_id)
            actual_ids = {int(v.object_id) for v in result.values}
            actual_surfaces: set[str] = set()
            for value_surface in expected[surface][family_name]:
                ids = runtime.resolve_surface_ids(value_surface, modality=Modality.TEXT.value, locale="")
                if len(ids) == 1 and ids[0] in actual_ids:
                    actual_surfaces.add(value_surface)
            ok = actual_surfaces == expected[surface][family_name] and len(actual_ids) == len(expected[surface][family_name])
            all_correct = all_correct and ok
            query_rows.append({
                "surface": surface,
                "relation": family_name,
                "status": result.status,
                "value_count": len(result.values),
                "correct": ok,
            })
        # IS_A proves scalar routing also survives the direct compile.
        scalar = runtime.resolve(sids[0], is_a_id)
        scalar_ok = scalar.status == "resolved" and len(scalar.values) == 1
        all_correct = all_correct and scalar_ok
        query_rows.append({
            "surface": surface, "relation": RelationType.IS_A.value,
            "status": scalar.status, "value_count": len(scalar.values), "correct": scalar_ok,
        })

    runtime_artifacts = sorted(p.name for p in out.iterdir() if p.suffix in {".json", ".eqbase"} and p.name != "report.json")
    report = {
        "format": DIRECT_TEACHER_FLAT_COMPILER_FORMAT,
        "version": DIRECT_TEACHER_FLAT_COMPILER_VERSION,
        "runtime_format_version": UNIFIED_NUMERIC_KERNEL_VERSION,
        "teacher": teacher_manifest,
        "teacher_state_bytes": teacher_state_bytes,
        "teacher_queries": int(structure.teacher_queries),
        "teacher_removed_before_runtime": not teacher_state.exists(),
        "forge_db_removed_before_runtime": not db_path.exists(),
        "structure": structure.to_dict(),
        "base_canonical": base.to_dict(),
        "relation_facets": facets.to_dict(),
        "closure": closure.to_dict(),
        "concept_macro": concept_macro.to_dict(),
        "relation_macro": relation_macro.to_dict(),
        "direct_compile": direct_summary.to_dict(),
        "flat_pack_bytes": final_path.stat().st_size,
        "legacy_oracle_core_parity": bool(parity),
        "query_accuracy": 1.0 if all_correct else 0.0,
        "query_rows": query_rows,
        "runtime_artifacts": runtime_artifacts,
        "runtime_artifact_count": len(runtime_artifacts),
        "controls": {
            "teacher_removed_before_runtime": not teacher_state.exists(),
            "forge_db_removed_before_runtime": not db_path.exists(),
            "single_runtime_artifact": runtime_artifacts == [final_path.name],
            "no_intermediate_runtime_pack_persisted": runtime_artifacts == [final_path.name],
            "direct_pack_matches_legacy_flattened_core": bool(parity),
            "runtime_queries_correct": bool(all_correct),
            "kernel_pack_is_loadable_after_forge_removal": loaded.digest == direct_pack.digest,
            "concept_macros_present": concept_macro.macro_count >= 3,
            "relation_macros_present": relation_macro.macro_count >= 3,
        },
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Equinox v0.56 direct Teacher → flat numeric kernel compiler trial.")
    ap.add_argument("--out", default="out/v0_56_direct_teacher_flat_compiler")
    ap.add_argument("--fixture", default=None)
    args = ap.parse_args(argv)
    report = run_direct_teacher_flat_compiler_trial(args.out, fixture_path=args.fixture)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if all(report["controls"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
