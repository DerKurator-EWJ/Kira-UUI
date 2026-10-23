from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
import argparse
import hashlib
import json
import sqlite3

from .canonical_id_space import (
    bind_surface,
    canonical_node_id,
    canonical_relation_id,
    load_numeric_triples,
    materialize_canonical_id_space,
)
from .ir import IRNode, Modality, NodeKind, RelationEvidence, RelationType, SourceRef
from .numeric_runtime import (
    NumericResolution,
    NumericRuntimeFact,
    NumericRuntimePack,
    NumericRuntimePolicy,
    NumericSurfaceBinding,
    SpecificToGeneralNumericRuntime,
    build_numeric_runtime_pack,
)
from .recursive_macro_collapse import (
    RecursiveMacroPolicy,
    effective_macro_facts,
    latest_recursive_macro_run_id,
    materialize_recursive_macro_collapse,
)
from .structural_closure import StructuralClosurePolicy, materialize_structural_closures

MACRO_NUMERIC_RUNTIME_FORMAT = "equinox-macro-aware-numeric-runtime-v1"
MACRO_NUMERIC_RUNTIME_VERSION = "0.49"


@dataclass(frozen=True)
class MacroShortcut:
    concept_id: int
    predicate_id: int
    macro_id: int
    object_id: int
    origin_subject_id: int
    original_inheritance_depth: int
    original_inheritance_path: tuple[int, ...]
    score: float
    confidence: float
    source_count: int
    evidence_count: int

    @property
    def quality(self) -> float:
        return max(0.0, min(1.0, float(self.score) * float(self.confidence)))

    def to_list(self) -> list[Any]:
        return [
            int(self.concept_id), int(self.predicate_id), int(self.macro_id), int(self.object_id),
            int(self.origin_subject_id), int(self.original_inheritance_depth),
            [int(x) for x in self.original_inheritance_path],
            float(self.score), float(self.confidence), int(self.source_count), int(self.evidence_count),
        ]

    @classmethod
    def from_list(cls, row: list[Any]) -> "MacroShortcut":
        if len(row) != 11:
            raise ValueError("macro shortcut row must have eleven fields")
        return cls(
            int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]),
            tuple(int(x) for x in row[6]), float(row[7]), float(row[8]), int(row[9]), int(row[10]),
        )


@dataclass
class MacroNumericRuntimePack:
    format: str
    version: str
    base: NumericRuntimePack
    recursive_macro_run_id: int
    recursive_macro_digest: str
    shortcuts: list[MacroShortcut]
    digest: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            "format": self.format,
            "version": self.version,
            "base": self.base.to_payload(include_digest=True),
            "recursive_macro_run_id": int(self.recursive_macro_run_id),
            "recursive_macro_digest": self.recursive_macro_digest,
            "shortcuts": [s.to_list() for s in self.shortcuts],
            "metadata": self.metadata,
        }
        if include_digest:
            payload["digest"] = self.digest
        return payload


@dataclass(frozen=True)
class MacroNumericResolution:
    resolution: NumericResolution
    route_kind: str
    macro_id: int | None
    nodes_touched: int
    baseline_nodes_touched: int
    shortcut_saved_nodes: int
    fallback_used: bool

    def to_dict(self) -> dict[str, Any]:
        out = self.resolution.to_dict()
        out.update({
            "route_kind": self.route_kind,
            "macro_id": self.macro_id,
            "nodes_touched": int(self.nodes_touched),
            "baseline_nodes_touched": int(self.baseline_nodes_touched),
            "shortcut_saved_nodes": int(self.shortcut_saved_nodes),
            "fallback_used": bool(self.fallback_used),
        })
        return out


def _digest(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _base_from_payload(payload: dict[str, Any]) -> NumericRuntimePack:
    # The base pack has already been digest-checked as part of the outer pack digest;
    # retain its own digest for auditability as well.
    return NumericRuntimePack(
        str(payload["format"]), str(payload["version"]), int(payload["canonical_run_id"]),
        str(payload["canonical_digest"]), tuple(int(x) for x in payload["inheritance_predicate_ids"]),
        [NumericRuntimeFact.from_list(x) for x in payload["relations"]],
        [NumericSurfaceBinding.from_list(x) for x in payload["surfaces"]],
        str(payload["digest"]), dict(payload.get("metadata") or {}),
    )


def _baseline_nodes_touched(runtime: SpecificToGeneralNumericRuntime, subject_id: int, predicate_id: int) -> int:
    """Count hierarchy nodes the v0.48 resolver must inspect before deciding."""
    count = 0
    pid = int(predicate_id)
    for level in runtime._levels(int(subject_id)):  # intentionally mirrors frozen v0.48 semantics
        count += len(level)
        found = False
        for origin, _ in level:
            if any(f.quality + 1e-15 >= runtime.policy.min_quality for f in runtime._by_subject_predicate.get((origin, pid), [])):
                found = True
        if found:
            break
    return count


def _best_macro_object(facts: Iterable[Any], predicate_id: int, *, min_quality: float, ambiguity_margin: float) -> tuple[int, float, float] | None:
    candidates = [f for f in facts if int(f.predicate_id) == int(predicate_id) and f.quality + 1e-15 >= min_quality]
    if not candidates:
        return None
    candidates.sort(key=lambda f: (-f.quality, -f.confidence, f.object_id))
    top = candidates[0]
    if len(candidates) > 1:
        second = candidates[1]
        if int(top.object_id) != int(second.object_id) and (top.quality - second.quality) < ambiguity_margin:
            return None
    return int(top.object_id), float(top.score), float(top.confidence)


def build_macro_numeric_runtime_pack(
    db: sqlite3.Connection,
    output_path: str | Path,
    *,
    canonical_run_id: int | None = None,
    recursive_macro_run_id: int | None = None,
    policy: NumericRuntimePolicy | None = None,
    metadata: dict[str, Any] | None = None,
) -> MacroNumericRuntimePack:
    """Build a DB-free macro-routing pack with compile-time semantic validation.

    A shortcut is emitted only when the nearest concept->macro route reproduces the
    frozen v0.48 resolver's object for that exact concept/predicate and the baseline
    answer is inherited (depth > 0). Runtime therefore saves hierarchy traversal
    without granting macros new semantic authority.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    base_path = out.with_suffix(out.suffix + ".base.tmp")
    base = build_numeric_runtime_pack(db, base_path, canonical_run_id=canonical_run_id, metadata={"embedded_in": MACRO_NUMERIC_RUNTIME_VERSION})
    try:
        base_path.unlink(missing_ok=True)
    except TypeError:  # py3.10 compatibility
        if base_path.exists():
            base_path.unlink()

    macro_rid = int(recursive_macro_run_id or latest_recursive_macro_run_id(db))
    row = db.execute(
        "SELECT digest FROM recursive_macro_collapse_runs WHERE recursive_macro_run_id=?",
        (macro_rid,),
    ).fetchone()
    if not row:
        raise KeyError(f"unknown recursive_macro_run_id={macro_rid}")
    macro_digest = str(row[0])

    runtime = SpecificToGeneralNumericRuntime(base, policy)
    effective_cache: dict[int, list[Any]] = {}
    shortcuts: list[MacroShortcut] = []
    concept_routes = [
        (int(member), int(macro))
        for member, macro in db.execute(
            """SELECT member_id,macro_id FROM recursive_macro_members
               WHERE recursive_macro_run_id=? AND member_kind='concept'
               ORDER BY member_id,macro_id""",
            (macro_rid,),
        ).fetchall()
    ]
    predicates = sorted({int(r.predicate_id) for r in base.relations if int(r.predicate_id) not in set(base.inheritance_predicate_ids)})

    for concept_id, macro_id in concept_routes:
        macro_facts = effective_cache.setdefault(
            macro_id,
            list(effective_macro_facts(db, macro_id, recursive_macro_run_id=macro_rid).values()),
        )
        for pid in predicates:
            baseline = runtime.resolve(concept_id, pid)
            if baseline.status != "resolved" or baseline.object_id is None or not baseline.inherited:
                continue
            candidate = _best_macro_object(
                macro_facts, pid,
                min_quality=runtime.policy.min_quality,
                ambiguity_margin=runtime.policy.ambiguity_margin,
            )
            if candidate is None or int(candidate[0]) != int(baseline.object_id):
                continue
            shortcuts.append(MacroShortcut(
                concept_id, pid, macro_id, int(baseline.object_id), int(baseline.origin_subject_id or concept_id),
                int(baseline.inheritance_depth or 0), tuple(int(x) for x in baseline.inheritance_path),
                float(baseline.score), float(baseline.confidence), int(baseline.source_count), int(baseline.evidence_count),
            ))

    shortcuts.sort(key=lambda s: (s.concept_id, s.predicate_id, s.macro_id, s.object_id))
    pack = MacroNumericRuntimePack(
        MACRO_NUMERIC_RUNTIME_FORMAT, MACRO_NUMERIC_RUNTIME_VERSION, base, macro_rid,
        macro_digest, shortcuts, "", dict(metadata or {}),
    )
    pack.digest = _digest(pack.to_payload(include_digest=False))
    out.write_text(json.dumps(pack.to_payload(include_digest=True), ensure_ascii=False, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return pack


def load_macro_numeric_runtime_pack(path: str | Path) -> MacroNumericRuntimePack:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != MACRO_NUMERIC_RUNTIME_FORMAT:
        raise ValueError(f"unsupported macro numeric runtime format: {payload.get('format')!r}")
    if payload.get("version") != MACRO_NUMERIC_RUNTIME_VERSION:
        raise ValueError(f"unsupported macro numeric runtime version: {payload.get('version')!r}")
    digest = str(payload.get("digest") or "")
    body = dict(payload)
    body.pop("digest", None)
    if digest != _digest(body):
        raise ValueError("macro numeric runtime pack digest mismatch")
    return MacroNumericRuntimePack(
        str(payload["format"]), str(payload["version"]), _base_from_payload(dict(payload["base"])),
        int(payload["recursive_macro_run_id"]), str(payload["recursive_macro_digest"]),
        [MacroShortcut.from_list(x) for x in payload["shortcuts"]], digest,
        dict(payload.get("metadata") or {}),
    )


class MacroAwareNumericRuntime:
    """v0.48-compatible resolution with validated macro-ID routing shortcuts."""

    def __init__(self, pack: MacroNumericRuntimePack, policy: NumericRuntimePolicy | None = None):
        self.pack = pack
        self.base_runtime = SpecificToGeneralNumericRuntime(pack.base, policy)
        self.policy = self.base_runtime.policy
        self._shortcuts: dict[tuple[int, int], list[MacroShortcut]] = {}
        for s in pack.shortcuts:
            self._shortcuts.setdefault((s.concept_id, s.predicate_id), []).append(s)
        for key in self._shortcuts:
            self._shortcuts[key].sort(key=lambda s: (-s.quality, s.macro_id, s.object_id))

    def resolve(self, subject_id: int, predicate_id: int) -> MacroNumericResolution:
        sid, pid = int(subject_id), int(predicate_id)
        baseline_work = _baseline_nodes_touched(self.base_runtime, sid, pid)

        # Preserve v0.48 local specificity exactly. Macro routing is only considered
        # if no qualifying local fact exists.
        local = [
            f for f in self.base_runtime._by_subject_predicate.get((sid, pid), [])
            if f.quality + 1e-15 >= self.policy.min_quality
        ]
        if local:
            base = self.base_runtime.resolve(sid, pid)
            return MacroNumericResolution(base, "local", None, 1, baseline_work, max(0, baseline_work - 1), False)

        routes = self._shortcuts.get((sid, pid), [])
        if routes:
            # Shortcuts were compile-time validated against v0.48. Multiple routes
            # are accepted only when they agree on the same object.
            object_ids = {s.object_id for s in routes}
            if len(object_ids) == 1:
                best = routes[0]
                base = NumericResolution(
                    "resolved", sid, pid, best.object_id, best.origin_subject_id,
                    best.original_inheritance_depth, best.original_inheritance_path,
                    best.score, best.confidence, best.source_count, best.evidence_count, (best.object_id,),
                    "validated macro-ID routing shortcut",
                )
                touched = 2  # concept local index + macro route
                return MacroNumericResolution(
                    base, "macro", best.macro_id, touched, baseline_work,
                    max(0, baseline_work - touched), False,
                )

        base = self.base_runtime.resolve(sid, pid)
        return MacroNumericResolution(base, "fallback", None, baseline_work, baseline_work, 0, True)

    def resolve_surface_query(self, surface: str, predicate_id: int, *, modality: str = Modality.TEXT.value, locale: str = "") -> MacroNumericResolution:
        ids = self.base_runtime.resolve_surface(surface, modality=modality, locale=locale)
        if len(ids) != 1:
            base = self.base_runtime.resolve_surface_query(surface, predicate_id, modality=modality, locale=locale)
            return MacroNumericResolution(base, "surface", None, 0, 0, 0, False)
        return self.resolve(ids[0], predicate_id)


def _node(name: str) -> IRNode:
    return IRNode(f"concept:{name}", Modality.CROSSMODAL.value, NodeKind.CONCEPT.value, 3, surface=name)


def run_macro_numeric_runtime_demo(out_dir: str | Path) -> dict[str, Any]:
    """Frozen v0.49 proof: same answers, fewer hierarchy-node touches."""
    from .database import CanonicalDB

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    db_path = out / "macro-numeric-runtime-demo.sqlite"
    base_pack_path = out / "baseline.eqnum.json"
    macro_pack_path = out / "macro.eqnum.json"
    if db_path.exists():
        db_path.unlink()

    names = [
        "owl", "falcon", "dog", "wolf", "bird", "mammal", "animal", "living", "root",
        "feathers", "fly", "fur", "walk", "alive", "move", "metabolize", "exists", "night",
    ]
    nodes = {x: _node(x) for x in names}
    src = SourceRef("v049-demo", "1", "macro-aware-numeric-runtime")
    ev = [
        RelationEvidence(nodes["owl"].node_id, nodes["bird"].node_id, RelationType.IS_A.value, 1, 1, src),
        RelationEvidence(nodes["falcon"].node_id, nodes["bird"].node_id, RelationType.IS_A.value, 1, 1, src),
        RelationEvidence(nodes["dog"].node_id, nodes["mammal"].node_id, RelationType.IS_A.value, 1, 1, src),
        RelationEvidence(nodes["wolf"].node_id, nodes["mammal"].node_id, RelationType.IS_A.value, 1, 1, src),
        RelationEvidence(nodes["bird"].node_id, nodes["animal"].node_id, RelationType.IS_A.value, 1, 1, src),
        RelationEvidence(nodes["mammal"].node_id, nodes["animal"].node_id, RelationType.IS_A.value, 1, 1, src),
        RelationEvidence(nodes["animal"].node_id, nodes["living"].node_id, RelationType.IS_A.value, 1, 1, src),
        RelationEvidence(nodes["living"].node_id, nodes["root"].node_id, RelationType.IS_A.value, 1, 1, src),
        RelationEvidence(nodes["bird"].node_id, nodes["feathers"].node_id, "covering", .99, .99, src),
        RelationEvidence(nodes["bird"].node_id, nodes["fly"].node_id, "flight", .98, .98, src),
        RelationEvidence(nodes["mammal"].node_id, nodes["fur"].node_id, "covering", .99, .99, src),
        RelationEvidence(nodes["mammal"].node_id, nodes["walk"].node_id, "locomotion", .98, .98, src),
        RelationEvidence(nodes["animal"].node_id, nodes["alive"].node_id, "alive_state", .99, .99, src),
        RelationEvidence(nodes["animal"].node_id, nodes["move"].node_id, "mobility", .99, .99, src),
        RelationEvidence(nodes["living"].node_id, nodes["metabolize"].node_id, "metabolism", .98, .98, src),
        RelationEvidence(nodes["root"].node_id, nodes["exists"].node_id, "existence", .97, .97, src),
        # leaf-local fact proves that macro routing never bypasses specificity.
        RelationEvidence(nodes["owl"].node_id, nodes["night"].node_id, "activity_period", .95, .95, src),
    ]

    with CanonicalDB(db_path) as cdb:
        for n in nodes.values():
            cdb.upsert_node(n)
        for e in ev:
            cdb.add_evidence(e)
        cdb.db.commit()
        canonical = materialize_canonical_id_space(cdb.db)
        ids = {name: canonical_node_id(cdb.db, node.node_id) for name, node in nodes.items()}
        for name in ("owl", "falcon", "dog", "wolf"):
            bind_surface(cdb.db, surface=name, concept_node_id=nodes[name].node_id, modality=Modality.TEXT.value, locale="en")
        cdb.db.commit()

        roots = [ids[x] for x in ("owl", "falcon", "dog", "wolf")]
        closure = materialize_structural_closures(
            cdb.db, canonical_run_id=canonical.run_id, root_ids=roots,
            policy=StructuralClosurePolicy(max_parent_depth=8),
        )
        macro_build = materialize_recursive_macro_collapse(
            cdb.db, canonical_run_id=canonical.run_id, structural_closure_run_id=closure.run_id,
            root_ids=roots, policy=RecursiveMacroPolicy(min_members=2, min_common_facts=2, max_levels=4),
        )
        base_pack = build_numeric_runtime_pack(cdb.db, base_pack_path, canonical_run_id=canonical.run_id)
        macro_pack = build_macro_numeric_runtime_pack(
            cdb.db, macro_pack_path, canonical_run_id=canonical.run_id,
            recursive_macro_run_id=macro_build.run_id,
        )
        predicates = {name: canonical_relation_id(cdb.db, name) for name in (
            "covering", "flight", "locomotion", "alive_state", "mobility",
            "metabolism", "existence", "activity_period",
        )}

    baseline = SpecificToGeneralNumericRuntime(base_pack)
    macro_rt = MacroAwareNumericRuntime(load_macro_numeric_runtime_pack(macro_pack_path))

    queries = [
        ("owl", predicates["covering"], ids["feathers"]),
        ("owl", predicates["alive_state"], ids["alive"]),
        ("owl", predicates["metabolism"], ids["metabolize"]),
        ("owl", predicates["existence"], ids["exists"]),
        ("owl", predicates["activity_period"], ids["night"]),
        ("falcon", predicates["flight"], ids["fly"]),
        ("falcon", predicates["existence"], ids["exists"]),
        ("dog", predicates["covering"], ids["fur"]),
        ("dog", predicates["alive_state"], ids["alive"]),
        ("dog", predicates["existence"], ids["exists"]),
        ("wolf", predicates["locomotion"], ids["walk"]),
        ("wolf", predicates["metabolism"], ids["metabolize"]),
    ]

    rows = []
    parity = True
    metadata_parity = True
    expected_ok = True
    baseline_work = 0
    macro_work = 0
    macro_routes = 0
    for name, pid, expected in queries:
        sid = ids[name]
        b = baseline.resolve(sid, pid)
        m = macro_rt.resolve(sid, pid)
        same = (b.status, b.object_id) == (m.resolution.status, m.resolution.object_id)
        same_metadata = (
            b.status == m.resolution.status
            and b.object_id == m.resolution.object_id
            and b.origin_subject_id == m.resolution.origin_subject_id
            and b.inheritance_depth == m.resolution.inheritance_depth
            and b.inheritance_path == m.resolution.inheritance_path
            and abs(b.score - m.resolution.score) < 1e-15
            and abs(b.confidence - m.resolution.confidence) < 1e-15
            and b.source_count == m.resolution.source_count
            and b.evidence_count == m.resolution.evidence_count
        )
        parity = parity and same
        metadata_parity = metadata_parity and same_metadata
        expected_ok = expected_ok and b.status == "resolved" and b.object_id == expected
        bw = _baseline_nodes_touched(baseline, sid, pid)
        baseline_work += bw
        macro_work += m.nodes_touched
        macro_routes += int(m.route_kind == "macro")
        rows.append({
            "subject": name, "predicate_id": pid, "expected_object_id": expected,
            "baseline": b.to_dict(), "macro": m.to_dict(), "same_answer": same,
            "same_resolution_metadata": same_metadata,
        })

    reduction = 0.0 if baseline_work == 0 else 1.0 - (macro_work / baseline_work)
    report = {
        "format": "equinox-forge-macro-aware-numeric-runtime-demo-v1",
        "version": MACRO_NUMERIC_RUNTIME_VERSION,
        "canonical_triple_count": len(base_pack.relations),
        "macro_levels": macro_build.levels_built,
        "macro_count": macro_build.macro_count,
        "shortcut_count": len(macro_pack.shortcuts),
        "queries": rows,
        "work": {
            "baseline_nodes_touched": baseline_work,
            "macro_nodes_touched": macro_work,
            "node_reduction_fraction": reduction,
            "macro_routed_queries": macro_routes,
            "query_count": len(queries),
        },
        "controls": {
            "answer_parity_with_v048": parity,
            "resolution_metadata_parity_with_v048": metadata_parity,
            "expected_answers_preserved": expected_ok,
            "positive_work_reduction": reduction > 0.0,
            "material_work_reduction": reduction >= 0.25,
            "macros_used_for_routing": macro_routes >= 6,
            "local_specificity_not_bypassed": any(r["subject"] == "owl" and r["macro"]["route_kind"] == "local" for r in rows),
            "runtime_pack_is_db_independent": True,
            "canonical_graph_not_rewritten": len(base_pack.relations) == len(ev),
        },
        "pack_digest": macro_pack.digest,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build and test the Equinox v0.49 macro-aware numeric runtime.")
    ap.add_argument("--out", default="out/v0_49_macro_numeric_runtime")
    args = ap.parse_args(argv)
    report = run_macro_numeric_runtime_demo(args.out)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
