from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import argparse
import gc
import hashlib
import json
import shutil

from .canonical_id_space import canonical_node_id, canonical_relation_id, materialize_canonical_id_space
from .database import CanonicalDB
from .hf_adapter import inspect_hf_model
from .ir import Modality, RelationType
from .macro_numeric_runtime import (
    MacroAwareNumericRuntime,
    build_macro_numeric_runtime_pack,
    load_macro_numeric_runtime_pack,
)
from .numeric_runtime import SpecificToGeneralNumericRuntime
from .real_teacher_domain_trial import _NeuralDomainTeacher
from .recursive_macro_collapse import RecursiveMacroPolicy, materialize_recursive_macro_collapse
from .structural_closure import StructuralClosurePolicy, materialize_structural_closures
from .top_down_teacher_structure import (
    AttributeProbeSpec,
    TopDownTeacherPolicy,
    _concept_node_id,
    _fixture_path,
    extract_top_down_teacher_structure,
)

TEACHER_NUMERIC_COMPILER_FORMAT = "equinox-teacher-numeric-compiler-trial-v1"
TEACHER_NUMERIC_COMPILER_VERSION = "0.50"


@dataclass(frozen=True)
class CompilerQueryCase:
    surface: str
    predicate: str
    expected_surface: str
    baseline_status: str
    baseline_object_id: int | None
    macro_status: str
    macro_object_id: int | None
    correct: bool
    route_kind: str
    baseline_nodes_touched: int
    macro_nodes_touched: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_fixture(path: str | Path | None = None) -> dict[str, Any]:
    return json.loads(Path(path or _fixture_path()).read_text(encoding="utf-8"))


def _policy(fixture: dict[str, Any]) -> TopDownTeacherPolicy:
    return TopDownTeacherPolicy(
        parent_prompt_template=str(fixture["parent_probe_template"]),
        attribute_probes=tuple(
            AttributeProbeSpec(str(x["relation_type"]), str(x["prompt_template"]))
            for x in fixture["attribute_probes"]
        ),
        stop_surfaces=tuple(str(x) for x in fixture["stop_surfaces"]),
        max_parent_depth=8,
        top_k=4,
        temperature=1.0,
        min_probability=0.55,
        evidence_confidence=0.95,
        extraction_namespace="v050-compiler-trial",
        attributes_on_internal_concepts_only=True,
    )


def _training_index(fixture: dict[str, Any]) -> dict[str, str]:
    return {str(prompt): str(target) for prompt, target in fixture["training_pairs"]}


def _expected_leaf_queries(fixture: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Queries that the scalar v0.48/v0.49 runtime is semantically able to answer.

    The generic PROPERTY/ACTION predicates are single-valued at the nearest
    specificity level.  This deliberately tests the nearest inherited value rather
    than pretending that v0.50 already exposes the complete multi-valued closure.
    """
    pairs = _training_index(fixture)
    parent = {str(a): str(b) for a, b in fixture["expected_parent_edges"]}
    rows: list[tuple[str, str, str]] = []
    for leaf in sorted(str(x) for x in fixture["seed_surfaces"]):
        immediate = parent[leaf]
        rows.append((leaf, RelationType.IS_A.value, immediate))
        prop = pairs.get(f"{immediate} hat")
        act = pairs.get(f"{immediate} kann")
        if prop:
            rows.append((leaf, RelationType.PROPERTY.value, prop))
        if act:
            rows.append((leaf, RelationType.ACTION.value, act))
    return rows


def _expected_teacher_triples(
    fixture: dict[str, Any],
    runtime: SpecificToGeneralNumericRuntime,
    predicate_ids: dict[str, int],
) -> set[tuple[int, int, int]]:
    expected: set[tuple[int, int, int]] = set()
    for prompt, target in fixture["training_pairs"]:
        prompt = str(prompt)
        subject = prompt.rsplit(" ", 2)[0]
        if prompt.endswith(" ist ein"):
            rel = RelationType.IS_A.value
            subject = prompt[: -len(" ist ein")]
        elif prompt.endswith(" hat"):
            rel = RelationType.PROPERTY.value
            subject = prompt[: -len(" hat")]
        elif prompt.endswith(" kann"):
            rel = RelationType.ACTION.value
            subject = prompt[: -len(" kann")]
        else:
            continue
        sids = runtime.resolve_surface(subject, modality=Modality.TEXT.value, locale="")
        oids = runtime.resolve_surface(str(target), modality=Modality.TEXT.value, locale="")
        if len(sids) == 1 and len(oids) == 1:
            expected.add((sids[0], predicate_ids[rel], oids[0]))
    return expected


def _static_hf_probe(model_dir: str | Path | None) -> dict[str, Any]:
    if not model_dir:
        return {
            "requested": False,
            "available": False,
            "note": "No external Hugging Face model directory was supplied to this environment.",
        }
    p = Path(model_dir)
    try:
        manifest = inspect_hf_model(p)
        return {
            "requested": True,
            "available": True,
            "model_dir": str(p.resolve()),
            "manifest": manifest.to_dict(),
            "runtime_execution_attempted": False,
            "note": "Static local HF manifest/tensor inspection succeeded. Causal execution requires the optional transformers runtime.",
        }
    except Exception as exc:
        return {"requested": True, "available": False, "model_dir": str(p), "error": repr(exc)}


def run_teacher_numeric_compiler_trial(
    out_dir: str | Path,
    *,
    fixture_path: str | Path | None = None,
    hf_model_dir: str | Path | None = None,
) -> dict[str, Any]:
    """End-to-end v0.50 trial over the new numeric Equinox path.

    The default benchmark uses a computed neural teacher so it is reproducible in an
    offline test environment.  An optional local Hugging Face directory is inspected
    through the existing HF adapter boundary, but this function does not claim an
    external pretrained-model result unless such a model is actually supplied.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fixture = _load_fixture(fixture_path)
    teacher = _NeuralDomainTeacher(
        [(str(a), str(b)) for a, b in fixture["training_pairs"]],
        out / "teacher",
        seed=5001,
        max_epochs=800,
    )
    teacher_manifest = teacher.manifest()
    teacher_state_path = Path(teacher_manifest["model_dir"])
    teacher_state_bytes = teacher_state_path.stat().st_size
    policy = _policy(fixture)

    db_path = out / "forge.sqlite"
    base_pack_path = out / "knowledge.eqnum.json"
    macro_pack_path = out / "knowledge.macro.eqnum.json"
    if db_path.exists():
        db_path.unlink()

    with CanonicalDB(db_path) as cdb:
        structure = extract_top_down_teacher_structure(
            cdb.db,
            teacher,
            seed_surfaces=fixture["seed_surfaces"],
            source_id="v050-neural-teacher",
            source_version="1",
            policy=policy,
            metadata={"trial": TEACHER_NUMERIC_COMPILER_FORMAT},
        )
        canonical = materialize_canonical_id_space(cdb.db)
        seed_numeric = {
            str(surface): canonical_node_id(
                cdb.db,
                _concept_node_id(policy.extraction_namespace, str(surface)),
            )
            for surface in fixture["seed_surfaces"]
        }
        closure = materialize_structural_closures(
            cdb.db,
            canonical_run_id=canonical.run_id,
            root_ids=seed_numeric.values(),
            policy=StructuralClosurePolicy(max_parent_depth=8),
        )
        macro = materialize_recursive_macro_collapse(
            cdb.db,
            canonical_run_id=canonical.run_id,
            structural_closure_run_id=closure.run_id,
            root_ids=seed_numeric.values(),
            policy=RecursiveMacroPolicy(min_members=2, min_common_facts=2, max_levels=4),
        )
        macro_pack = build_macro_numeric_runtime_pack(
            cdb.db,
            macro_pack_path,
            canonical_run_id=canonical.run_id,
            recursive_macro_run_id=macro.run_id,
            metadata={"compiler_trial": TEACHER_NUMERIC_COMPILER_VERSION},
        )
        # Also persist the embedded base as a standalone artifact for size/parity audit.
        base_pack_path.write_text(
            json.dumps(macro_pack.base.to_payload(include_digest=True), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        predicate_ids = {
            rel: canonical_relation_id(cdb.db, rel)
            for rel in (RelationType.IS_A.value, RelationType.PROPERTY.value, RelationType.ACTION.value)
        }
        canonical_triple_count = len(macro_pack.base.relations)
        canonical_digest = canonical.digest
        macro_digest = macro.digest

    # Teacher OFF: the runtime evaluation starts only after DB closure and teacher
    # state removal from its expected path.  The disabled file is retained for byte
    # accounting/audit, but the runtime has no path to it.
    disabled_teacher_path = teacher_state_path.with_suffix(teacher_state_path.suffix + ".disabled")
    if disabled_teacher_path.exists():
        disabled_teacher_path.unlink()
    shutil.move(str(teacher_state_path), str(disabled_teacher_path))
    del teacher
    gc.collect()

    loaded_macro_pack = load_macro_numeric_runtime_pack(macro_pack_path)
    macro_rt = MacroAwareNumericRuntime(loaded_macro_pack)
    baseline_rt = SpecificToGeneralNumericRuntime(loaded_macro_pack.base)

    expected_triples = _expected_teacher_triples(fixture, baseline_rt, predicate_ids)
    actual_triples = {
        (int(r.subject_id), int(r.predicate_id), int(r.object_id))
        for r in loaded_macro_pack.base.relations
    }
    stored_teacher_fact_retention = (
        len(expected_triples & actual_triples) / len(expected_triples) if expected_triples else 0.0
    )

    query_rows: list[CompilerQueryCase] = []
    baseline_work = 0
    macro_work = 0
    macro_routes = 0
    for surface, predicate, expected_surface in _expected_leaf_queries(fixture):
        p = predicate_ids[predicate]
        sids = baseline_rt.resolve_surface(surface, modality=Modality.TEXT.value, locale="")
        expected_ids = baseline_rt.resolve_surface(expected_surface, modality=Modality.TEXT.value, locale="")
        if len(sids) != 1 or len(expected_ids) != 1:
            raise AssertionError((surface, expected_surface, sids, expected_ids))
        b = baseline_rt.resolve(sids[0], p)
        m = macro_rt.resolve(sids[0], p)
        ok = b.status == "resolved" and m.resolution.status == "resolved" and b.object_id == expected_ids[0] == m.resolution.object_id
        baseline_work += int(m.baseline_nodes_touched)
        macro_work += int(m.nodes_touched)
        macro_routes += int(m.route_kind == "macro")
        query_rows.append(CompilerQueryCase(
            surface,
            predicate,
            expected_surface,
            b.status,
            b.object_id,
            m.resolution.status,
            m.resolution.object_id,
            bool(ok),
            m.route_kind,
            int(m.baseline_nodes_touched),
            int(m.nodes_touched),
        ))

    scalar_accuracy = sum(x.correct for x in query_rows) / len(query_rows) if query_rows else 0.0
    reduction = 0.0 if baseline_work == 0 else 1.0 - (macro_work / baseline_work)
    macro_pack_bytes = macro_pack_path.stat().st_size
    base_pack_bytes = base_pack_path.stat().st_size
    hf_probe = _static_hf_probe(hf_model_dir)

    # v0.50 intentionally audits a remaining semantic limitation: generic PROPERTY
    # and ACTION predicates are scalar in the v0.48 resolver.  The numeric pack keeps
    # every extracted relation, but one scalar query returns only the nearest value.
    relation_value_counts: dict[str, int] = {}
    for rel_name, pid in predicate_ids.items():
        relation_value_counts[rel_name] = len({r.object_id for r in loaded_macro_pack.base.relations if r.predicate_id == pid})
    scalar_multivalue_limitation_present = any(
        relation_value_counts.get(rel, 0) > 1 for rel in (RelationType.PROPERTY.value, RelationType.ACTION.value)
    )

    report = {
        "format": TEACHER_NUMERIC_COMPILER_FORMAT,
        "version": TEACHER_NUMERIC_COMPILER_VERSION,
        "scope": "offline-computed-neural-teacher-end-to-end-numeric-compiler",
        "external_pretrained_hf_claim": bool(hf_probe.get("requested") and hf_probe.get("available") and hf_probe.get("runtime_execution_attempted")),
        "teacher": teacher_manifest,
        "teacher_state_bytes": teacher_state_bytes,
        "teacher_disabled_before_runtime": not teacher_state_path.exists() and disabled_teacher_path.exists(),
        "teacher_queries": structure.teacher_queries,
        "teacher_structure": structure.to_dict(),
        "canonical": canonical.to_dict(),
        "closure": closure.to_dict(),
        "macro": macro.to_dict(),
        "canonical_triple_count": canonical_triple_count,
        "expected_teacher_structural_facts": len(expected_triples),
        "stored_teacher_structural_facts": len(expected_triples & actual_triples),
        "stored_teacher_fact_retention": stored_teacher_fact_retention,
        "scalar_runtime_query_accuracy": scalar_accuracy,
        "scalar_runtime_cases": [x.to_dict() for x in query_rows],
        "work": {
            "baseline_nodes_touched": baseline_work,
            "macro_nodes_touched": macro_work,
            "node_reduction_fraction": reduction,
            "macro_routed_queries": macro_routes,
            "macro_shortcut_count": len(loaded_macro_pack.shortcuts),
            "query_count": len(query_rows),
        },
        "artifacts": {
            "base_pack_bytes": base_pack_bytes,
            "macro_pack_bytes": macro_pack_bytes,
            "macro_pack_to_teacher_state_ratio": macro_pack_bytes / teacher_state_bytes if teacher_state_bytes else None,
            "canonical_digest": canonical_digest,
            "macro_digest": macro_digest,
            "runtime_pack_digest": loaded_macro_pack.digest,
        },
        "hf_local_model_probe": hf_probe,
        "relation_value_counts": relation_value_counts,
        "known_limitations": {
            "generic_property_action_are_scalar_queries": scalar_multivalue_limitation_present,
            "coarse_predicates_block_safe_macro_shortcuts": scalar_multivalue_limitation_present and len(loaded_macro_pack.shortcuts) == 0,
            "full_relational_storage_is_preserved_despite_scalar_query_limit": stored_teacher_fact_retention >= 0.999999,
            "external_pretrained_hf_runtime_not_validated_here": not bool(hf_probe.get("runtime_execution_attempted")),
        },
        "controls": {
            "teacher_is_computed_neural_model": int(teacher_manifest.get("parameters", 0)) > 0,
            "teacher_disabled_before_runtime": not teacher_state_path.exists() and disabled_teacher_path.exists(),
            "forge_db_closed_before_runtime": True,
            "all_teacher_structural_facts_survive_numeric_pack": stored_teacher_fact_retention >= 0.999999,
            "scalar_specific_to_general_queries_correct": scalar_accuracy >= 0.999999,
            "recursive_macro_collapse_present": macro.levels_built >= 2 and macro.macro_count >= 3,
            "unsafe_ambiguous_macro_shortcuts_not_emitted": macro_routes == 0 and len(loaded_macro_pack.shortcuts) == 0,
            "macro_runtime_preserves_baseline_when_shortcut_is_unsafe": baseline_work == macro_work,
            "runtime_pack_is_smaller_than_teacher_state_in_this_fixture": macro_pack_bytes < teacher_state_bytes,
            "canonical_identity_is_numeric": all(isinstance(v, int) for r in loaded_macro_pack.base.relations for v in (r.subject_id, r.predicate_id, r.object_id)),
        },
        "notes": [
            "This is the first end-to-end trial of the new v0.44-v0.49 numeric architecture.",
            "The default teacher is a real locally trained neural model so the benchmark remains offline and reproducible.",
            "No external pretrained Hugging Face execution result is claimed unless a compatible local model/runtime is explicitly supplied.",
            "The runtime evaluation is performed after the Forge DB is closed and the teacher state is moved out of its original path.",
            "All extracted structural facts survive in the numeric pack; v0.50 separately exposes that generic PROPERTY/ACTION remain scalar query predicates in the current runtime API.",
            "Because the macro contains multiple values under those coarse predicates, v0.49 correctly refuses to emit an ambiguous shortcut; this leaves macro routing at zero in this end-to-end teacher fixture.",
        ],
    }
    (out / "compiler-trial-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Equinox Forge v0.50 teacher-to-numeric compiler trial")
    ap.add_argument("--out", default="out/v0_50_teacher_numeric_compiler")
    ap.add_argument("--fixture", default=None)
    ap.add_argument("--hf-model-dir", default=None, help="optional local Hugging Face model directory for static manifest/tensor inspection")
    args = ap.parse_args(argv)
    report = run_teacher_numeric_compiler_trial(args.out, fixture_path=args.fixture, hf_model_dir=args.hf_model_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if all(report["controls"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
