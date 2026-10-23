from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import argparse
import gc
import json
import shutil

from .canonical_id_space import canonical_node_id, canonical_relation_id, materialize_canonical_id_space
from .canonical_relation_facets import (
    FacetedMacroNumericRuntime,
    build_faceted_macro_runtime_pack,
    load_faceted_macro_runtime_pack,
    materialize_relation_faceted_id_space,
)
from .database import CanonicalDB
from .ir import Modality, RelationType
from .macro_numeric_runtime import build_macro_numeric_runtime_pack
from .real_teacher_domain_trial import _NeuralDomainTeacher
from .recursive_macro_collapse import RecursiveMacroPolicy, materialize_recursive_macro_collapse
from .structural_closure import StructuralClosurePolicy, materialize_structural_closures
from .teacher_numeric_compiler import _load_fixture, _policy
from .top_down_teacher_structure import _concept_node_id, extract_top_down_teacher_structure

RELATION_FACET_COMPILER_FORMAT = "equinox-relation-facet-compiler-trial-v1"
RELATION_FACET_COMPILER_VERSION = "0.51"


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
        out[leaf] = {
            RelationType.PROPERTY.value: props,
            RelationType.ACTION.value: actions,
        }
    return out


def run_relation_facet_compiler_trial(
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
        seed=5101,
        max_epochs=800,
    )
    teacher_manifest = teacher.manifest()
    teacher_state_path = Path(teacher_manifest["model_dir"])
    policy = _policy(fixture)

    db_path = out / "forge.sqlite"
    macro_pack_path = out / "knowledge.faceted.macro.eqnum.json"
    final_pack_path = out / "knowledge.faceted.eqnum.json"
    if db_path.exists():
        db_path.unlink()

    with CanonicalDB(db_path) as cdb:
        structure = extract_top_down_teacher_structure(
            cdb.db,
            teacher,
            seed_surfaces=fixture["seed_surfaces"],
            source_id="v051-neural-teacher",
            source_version="1",
            policy=policy,
            metadata={"trial": RELATION_FACET_COMPILER_FORMAT},
        )
        base_canonical = materialize_canonical_id_space(cdb.db)
        facets = materialize_relation_faceted_id_space(
            cdb.db,
            source_canonical_run_id=base_canonical.run_id,
            metadata={"trial": RELATION_FACET_COMPILER_FORMAT},
        )
        seed_numeric = {
            str(surface): canonical_node_id(
                cdb.db,
                _concept_node_id(policy.extraction_namespace, str(surface)),
            )
            for surface in fixture["seed_surfaces"]
        }
        closure = materialize_structural_closures(
            cdb.db,
            canonical_run_id=facets.derived_canonical_run_id,
            root_ids=seed_numeric.values(),
            policy=StructuralClosurePolicy(max_parent_depth=8),
        )
        macro = materialize_recursive_macro_collapse(
            cdb.db,
            canonical_run_id=facets.derived_canonical_run_id,
            structural_closure_run_id=closure.run_id,
            root_ids=seed_numeric.values(),
            policy=RecursiveMacroPolicy(min_members=2, min_common_facts=2, max_levels=4),
        )
        macro_pack = build_macro_numeric_runtime_pack(
            cdb.db,
            macro_pack_path,
            canonical_run_id=facets.derived_canonical_run_id,
            recursive_macro_run_id=macro.run_id,
            metadata={"relation_facets": True},
        )
        final_pack = build_faceted_macro_runtime_pack(
            cdb.db,
            final_pack_path,
            macro_pack=macro_pack,
            relation_facet_run_id=facets.run_id,
            metadata={"compiler_trial": RELATION_FACET_COMPILER_VERSION},
        )
        family_ids = {
            RelationType.PROPERTY.value: canonical_relation_id(cdb.db, RelationType.PROPERTY.value),
            RelationType.ACTION.value: canonical_relation_id(cdb.db, RelationType.ACTION.value),
        }
        facet_rows = cdb.db.execute(
            """SELECT facet_predicate_id,family_predicate_id,object_id
               FROM canonical_relation_facets WHERE relation_facet_run_id=?
               ORDER BY family_predicate_id,object_id""",
            (facets.run_id,),
        ).fetchall()
        relation_hierarchy_count = facets.relation_hierarchy_triple_count

    disabled_teacher_path = teacher_state_path.with_suffix(teacher_state_path.suffix + ".disabled")
    if disabled_teacher_path.exists():
        disabled_teacher_path.unlink()
    shutil.move(str(teacher_state_path), str(disabled_teacher_path))
    del teacher
    gc.collect()
    try:
        db_path.unlink()
    except FileNotFoundError:
        pass

    runtime = FacetedMacroNumericRuntime(load_faceted_macro_runtime_pack(final_pack_path))
    expected = _expected_family_values(fixture)
    family_rows: list[dict[str, Any]] = []
    all_correct = True
    macro_routes = 0
    total_values = 0
    baseline_work = 0
    macro_work = 0

    for surface in sorted(expected):
        ids = runtime.resolve_surface(surface, modality=Modality.TEXT.value, locale="")
        if len(ids) != 1:
            raise AssertionError((surface, ids))
        for family_name in (RelationType.PROPERTY.value, RelationType.ACTION.value):
            result = runtime.resolve_family(ids[0], family_ids[family_name])
            actual_surfaces: set[str] = set()
            actual_ids = {v.object_id for v in result.values}
            for value_surface in expected[surface][family_name]:
                value_ids = runtime.resolve_surface(value_surface, modality=Modality.TEXT.value, locale="")
                if len(value_ids) == 1 and value_ids[0] in actual_ids:
                    actual_surfaces.add(value_surface)
            ok = actual_surfaces == expected[surface][family_name] and len(actual_ids) == len(expected[surface][family_name])
            all_correct = all_correct and ok
            macro_routes += sum(1 for v in result.values if v.route_kind == "macro")
            total_values += len(result.values)
            baseline_work += int(result.baseline_nodes_touched)
            macro_work += int(result.nodes_touched)
            family_rows.append({
                "surface": surface,
                "family": family_name,
                "expected_values": sorted(expected[surface][family_name]),
                "resolved_value_count": len(result.values),
                "correct": ok,
                "routes": [v.route_kind for v in result.values],
                "inheritance_depths": [v.inheritance_depth for v in result.values],
            })

    relation_ids_first_class = all(
        isinstance(a, int) and isinstance(b, int) and isinstance(c, int)
        for a, b, c in facet_rows
    )
    shortcut_count = len(final_pack.macro_pack.shortcuts)
    reduction = 0.0 if baseline_work == 0 else 1.0 - (macro_work / baseline_work)
    report = {
        "format": RELATION_FACET_COMPILER_FORMAT,
        "version": RELATION_FACET_COMPILER_VERSION,
        "teacher": teacher_manifest,
        "teacher_disabled_before_runtime": not teacher_state_path.exists() and disabled_teacher_path.exists(),
        "forge_db_removed_before_runtime": not db_path.exists(),
        "teacher_queries": structure.teacher_queries,
        "base_canonical": base_canonical.to_dict(),
        "relation_facet_build": facets.to_dict(),
        "closure": closure.to_dict(),
        "macro": macro.to_dict(),
        "facet_runtime_pack_digest": final_pack.digest,
        "relation_facet_count": len(facet_rows),
        "relation_hierarchy_triple_count": relation_hierarchy_count,
        "family_query_accuracy": 1.0 if all_correct else 0.0,
        "family_query_rows": family_rows,
        "work": {
            "resolved_values": total_values,
            "macro_routed_values": macro_routes,
            "macro_shortcut_count": shortcut_count,
            "baseline_nodes_touched": baseline_work,
            "macro_nodes_touched": macro_work,
            "node_reduction_fraction": reduction,
        },
        "controls": {
            "relation_ids_are_first_class_numeric_objects": relation_ids_first_class,
            "relation_ids_have_numeric_family_membership": relation_hierarchy_count == len(facet_rows),
            "multi_valued_property_action_closure_is_lossless": all_correct,
            "teacher_off_before_runtime": not teacher_state_path.exists() and disabled_teacher_path.exists(),
            "forge_db_off_before_runtime": not db_path.exists(),
            "recursive_macro_collapse_survives_faceting": macro.levels_built >= 2 and macro.macro_count >= 3,
            "coarse_predicate_ambiguity_no_longer_blocks_all_shortcuts": shortcut_count > 0,
            "macro_routing_is_actually_used": macro_routes > 0,
            "macro_routing_reduces_hierarchy_work": reduction > 0.0,
        },
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Equinox v0.51 canonical relation-ID / multi-value closure trial.")
    ap.add_argument("--out", default="out/v0_51_relation_facets")
    ap.add_argument("--fixture", default=None)
    args = ap.parse_args(argv)
    report = run_relation_facet_compiler_trial(args.out, fixture_path=args.fixture)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
