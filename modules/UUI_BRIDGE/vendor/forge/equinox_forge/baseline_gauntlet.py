from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import argparse
import hashlib
import json
import shutil
import time

from .canonical import make_text_surface
from .compiler import CompilePolicy
from .database import CanonicalDB
from .exception_consensus import ExceptionConsensusPolicy, list_exception_states, track_exception_run
from .exception_induction import ExceptionInductionPolicy, induce_exception_overrides
from .frame_induction import FrameInductionPolicy, induce_relation_frames, persist_frame_document
from .frame_schema_benchmark import FrameEvalCase, compile_and_benchmark_frame_schemas
from .generalization import _pass_through_compression
from .generalization_ladder import (
    LadderEvalCase,
    LadderTrainCase,
    compile_and_benchmark_generalization_ladder,
)
from .hidden_state import HiddenStateConfig, HiddenStateSample, extract_contextual_hidden_evidence, load_probe_dataset
from .hierarchical_frame_transfer import HierarchicalFrameTransferPolicy, promote_frames_to_higher_concepts
from .ir import ForgeDocument, IRNode, Modality, NodeKind, RelationEvidence, RelationType, SourceRef
from .macro_discovery import MacroDiscoveryPolicy
from .prototype_hierarchy import PrototypeHierarchyPolicy, induce_prototype_hierarchy
from .runtime_cognition import RuntimeCognition, RuntimeCognitionPolicy
from .runtime_compression import MacroPromotionPolicy, compile_runtime_compression
from .runtime_pack import compile_base_pack, load_base_pack
from .runtime_query import benchmark_runtime_pack
from .sense_compiler import FusionWeights, SenseCompilePolicy, compile_provisional_senses, find_text_surface_id
from .teacher_behavior import BehaviourProbeConfig, TeacherCandidate, extract_teacher_behaviour
from .validate import validate_document

GAUNTLET_FORMAT = "equinox-forge-baseline-gauntlet-v1"
GAUNTLET_VERSION = "0.31"

# Frozen v0.31 deterministic case set. Benchmark semantics may be extended in later
# releases but these rows must not be silently rewritten.
GAUNTLET_DATA: dict[str, Any] = {
    "generalization": {
        "train": [
            ["Hund ist ein", "taxonomy"],
            ["Katze ist ein", "taxonomy"],
            ["Hund kann", "ability"],
            ["Katze kann", "ability"],
            ["Katze jagt Maus", "hunting"],
            ["Hund sieht Katze", "perception"],
            ["Haus ist", "property"],
            ["Vogel fliegt", "flight"],
            ["Katze wohnt", "residence-bare"],
            ["Vogel fliegt in", "motion-preposition"],
        ],
        "eval": [
            ["Hund kann", 0, "ability", "exact replay control"],
            ["Mein Hund kann", 1, "ability", "new prefix, known predictive suffix"],
            ["Hund jagt Maus", 2, "hunting", "known words + known local relation in new composition"],
            ["Maus sieht Hund", 3, "perception-reversed", "known lexicon, unseen local order"],
            ["Wolf ist ein", 4, "taxonomy", "novel entity in a known relational frame"],
            ["Hund wohnt in", 5, "residence-preposition", "entire relation family held out"],
        ],
        "abstention": [
            ["Stein singt", "all_unknown"],
            ["Quasar verdampft", "all_unknown"],
            ["Hund Baum", "known_lexicon_unsupported"],
            ["Katze fliegt", "known_lexicon_unsupported"],
            ["Haus kann", "known_lexicon_unsupported"],
            ["Maus kann", "known_lexicon_unsupported"],
            ["Stein ist ein", "mixed_unknown_known_frame"],
            ["Hund wohnt in", "heldout_relation_family"],
        ],
    },
    "typed_frame": {
        "train": ["Hund ist ein", "Katze ist ein"],
        "eval": [
            ["Wolf ist ein", True, "semantically admitted novel filler"],
            ["Fuchs ist ein", True, "second admitted novel filler"],
            ["Mein Wolf ist ein", True, "prefix transfer through typed frame"],
            ["Stein ist ein", False, "structural match but semantically inadmissible filler"],
        ],
    },
    "ambiguity": {
        "target": "Bank",
        "expected_clusters": 2,
        "expected_contexts": 6,
    },
    "exception_consensus": {
        "stable_min_sources": 2,
        "stable_min_runs": 2,
        "stable_consensus": 0.75,
        "quarantine_conflicts": 2,
        "expected_lifecycle": ["candidate", "stable", "contested", "quarantined", "candidate", "stable"],
    },
    "runtime_routing": {
        "parents": 80,
        "macro": ["a", "b", "c"],
    },
}


def gauntlet_dataset_sha256() -> str:
    raw = json.dumps(GAUNTLET_DATA, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class ScriptedLadderTeacher:
    backend_id = "v031-scripted-ladder"

    def manifest(self) -> dict[str, Any]:
        return {"backend_id": self.backend_id, "fixture": True, "frozen_dataset": gauntlet_dataset_sha256()}

    def next_token(self, prompt: str, top_k: int, temperature: float):
        p = prompt.strip()
        exact = {
            "Hund kann": [(" bellen", .78), (" laufen", .14), (" schlafen", .08)],
            "Mein Hund kann": [(" bellen", .78), (" laufen", .14), (" schlafen", .08)],
            "Hund jagt Maus": [(" heute", .72), (" schnell", .18), (" draußen", .10)],
            "Maus sieht Hund": [(" heute", .72), (" ruhig", .18), (" draußen", .10)],
            "Wolf ist ein": [(" Tier", .82), (" Wesen", .12), (" Haustier", .06)],
            "Hund wohnt in": [(" Haus", .80), (" Stadt", .12), (" Nähe", .08)],
            "Hund ist ein": [(" Tier", .82), (" Haustier", .12), (" Wesen", .06)],
            "Katze ist ein": [(" Tier", .82), (" Haustier", .12), (" Wesen", .06)],
            "Katze kann": [(" miauen", .78), (" schlafen", .14), (" laufen", .08)],
            "Katze jagt Maus": [(" heute", .72), (" schnell", .18), (" draußen", .10)],
            "Hund sieht Katze": [(" heute", .72), (" ruhig", .18), (" draußen", .10)],
            "Haus ist": [(" groß", .76), (" alt", .16), (" nah", .08)],
            "Vogel fliegt": [(" hoch", .76), (" schnell", .16), (" weit", .08)],
            "Katze wohnt": [(" hier", .76), (" ruhig", .16), (" allein", .08)],
            "Vogel fliegt in": [(" Luft", .76), (" Höhe", .16), (" Ferne", .08)],
        }
        vals = exact[p][:top_k]
        return [TeacherCandidate(i + 1, s, s, prob, 4.0 - i) for i, (s, prob) in enumerate(vals)]


class ScriptedTypedSlotTeacher:
    backend_id = "v031-scripted-typed-slot"

    def manifest(self) -> dict[str, Any]:
        return {"backend_id": self.backend_id, "fixture": True}

    def next_token(self, prompt: str, top_k: int, temperature: float):
        table = {
            "Hund ist ein": [(" Tier", .82), (" Haustier", .12), (" Wesen", .06)],
            "Katze ist ein": [(" Tier", .82), (" Haustier", .12), (" Wesen", .06)],
            "Wolf ist ein": [(" Tier", .84), (" Wesen", .10), (" Raubtier", .06)],
            "Fuchs ist ein": [(" Tier", .84), (" Wesen", .10), (" Raubtier", .06)],
            "Mein Wolf ist ein": [(" Tier", .84), (" Wesen", .10), (" Raubtier", .06)],
            "Stein ist ein": [(" Objekt", .80), (" Material", .12), (" Ding", .08)],
        }
        vals = table[prompt.strip()][:top_k]
        return [TeacherCandidate(i + 1, s, s, prob, 4.0 - i) for i, (s, prob) in enumerate(vals)]


class BaselineTeacher:
    backend_id = "v031-base"

    def manifest(self):
        return {"backend_id": self.backend_id, "fixture": True}

    def next_token(self, prompt, top_k, temperature):
        table = {
            "Hund atmet": [(" Luft", .92), (" Sauerstoff", .08)],
            "Katze atmet": [(" Luft", .92), (" Sauerstoff", .08)],
            "Adler fliegt": [(" Himmel", .91), (" hoch", .09)],
            "Falke fliegt": [(" Himmel", .91), (" hoch", .09)],
        }
        vals = table[prompt.strip()][:top_k]
        return [TeacherCandidate(i + 1, s, s, p, 3.0 - i) for i, (s, p) in enumerate(vals)]


class ExceptionTeacher:
    def __init__(self, support: bool, name: str):
        self.support = support
        self.backend_id = name

    def manifest(self):
        return {"backend_id": self.backend_id, "fixture": True, "supports_exception": self.support}

    def next_token(self, prompt, top_k, temperature):
        vals = [(" Sauerstoff", .86), (" Luft", .14)] if self.support else [(" Luft", .90), (" Sauerstoff", .10)]
        return [TeacherCandidate(100 + i, s, s, p, 4.0 - i) for i, (s, p) in enumerate(vals[:top_k])]


class ScriptedHiddenBackend:
    backend_id = "v031-scripted-hidden"

    def manifest(self):
        return {"backend_id": self.backend_id, "dimension": 4, "fixture": True}

    def contextual_state(self, text, target_start, target_end, layers, pooling):
        lo = text.lower()
        if any(k in lo for k in ["geld", "kredit", "konto", "überweist"]):
            base = [1.0, 0.04, 0.0, 0.0]
        else:
            base = [0.04, 1.0, 0.0, 0.0]
        return [HiddenStateSample(layer=l, vector=base, token_count=1, metadata={"fixture": True}) for l in layers]


def _typed_semantic_support_doc() -> ForgeDocument:
    nodes: dict[str, IRNode] = {}
    surfaces: dict[str, IRNode] = {}
    for text in ["Hund", "Katze", "Wolf", "Fuchs", "Stein"]:
        qs, node = make_text_surface(text)
        for q in qs:
            nodes[q.node_id] = q
        nodes[node.node_id] = node
        surfaces[text] = node
    src = SourceRef("v031-independent-typed-semantic", "1", "semantic_similarity_fixture")
    pairs = [
        ("Hund", "Katze", .92),
        ("Wolf", "Hund", .94), ("Wolf", "Katze", .90),
        ("Fuchs", "Hund", .91), ("Fuchs", "Katze", .89),
        ("Stein", "Hund", .08), ("Stein", "Katze", .06),
    ]
    evidence = [
        RelationEvidence(surfaces[a].node_id, surfaces[b].node_id, RelationType.SEMANTIC_SIMILARITY.value,
                         score, .98, src, {"independent_of_frame_teacher": True})
        for a, b, score in pairs
    ]
    return ForgeDocument(nodes=sorted(nodes.values(), key=lambda n: (n.level, n.node_id)), evidence=evidence)


def _exception_semantic_doc() -> ForgeDocument:
    names = ["Hund", "Katze", "Adler", "Falke", "Eule", "Stein"]
    nodes: dict[str, IRNode] = {}
    surfaces: dict[str, IRNode] = {}
    for text in names:
        qs, node = make_text_surface(text)
        for q in qs:
            nodes[q.node_id] = q
        nodes[node.node_id] = node
        surfaces[text] = node
    src = SourceRef("v031-independent-exception-semantic", "1", "independent_semantic_fixture")
    pairs = [
        ("Hund", "Katze", .98), ("Adler", "Falke", .98),
        ("Hund", "Adler", .91), ("Hund", "Falke", .90), ("Katze", "Adler", .90), ("Katze", "Falke", .91),
        ("Eule", "Adler", .97), ("Eule", "Falke", .96), ("Eule", "Hund", .90), ("Eule", "Katze", .90),
        ("Stein", "Hund", .05), ("Stein", "Katze", .04), ("Stein", "Adler", .03), ("Stein", "Falke", .04),
    ]
    evidence = [
        RelationEvidence(surfaces[a].node_id, surfaces[b].node_id, RelationType.SEMANTIC_SIMILARITY.value,
                         score, .99, src, {"independent": True})
        for a, b, score in pairs
    ]
    return ForgeDocument(nodes=sorted(nodes.values(), key=lambda n: (n.level, n.node_id)), evidence=evidence)


def _ingest(db: CanonicalDB, doc: ForgeDocument) -> None:
    for n in doc.nodes:
        db.upsert_node(n)
    for e in doc.evidence:
        db.add_evidence(e)
    db.commit()


def _top1(cog: RuntimeCognition, prompt: str) -> str | None:
    report = cog.predict(prompt)
    return report.predictions[0].surface if report.predictions else None


def run_generalization_and_rejection(
    out: Path,
    cognition_policy: RuntimeCognitionPolicy | None = None,
) -> dict[str, Any]:
    cfg = BehaviourProbeConfig(top_k=3, min_probability=0.0, evidence_confidence=0.95)
    # No supplied policy means the *historical v0.31 baseline semantics*. This
    # keeps eqforge-gauntlet reproducible after later runtime releases change
    # defaults. New releases may pass their current policy against the same data.
    policy = cognition_policy or RuntimeCognitionPolicy(
        max_recall_nodes=96, max_recall_depth=3, max_predictions=3, min_prediction_score=0.0,
        enable_support_gated_recalled_prediction=False,
    )
    train = [LadderTrainCase(*row) for row in GAUNTLET_DATA["generalization"]["train"]]
    evaluation = [LadderEvalCase(*row) for row in GAUNTLET_DATA["generalization"]["eval"]]
    ladder = compile_and_benchmark_generalization_ladder(
        ScriptedLadderTeacher(), train, evaluation, out / "ladder", "v031-ladder", "1",
        probe_cfg=cfg, cognition_policy=policy, runtime_top_k=3, pack_id="v031-ladder", pack_version=31,
    )
    pack_path = out / "ladder" / "train.eqbase"
    cog = RuntimeCognition(load_base_pack(pack_path), policy)
    rejection_rows = []
    for prompt, category in GAUNTLET_DATA["generalization"]["abstention"]:
        pred = cog.predict(prompt)
        rejection_rows.append({
            "prompt": prompt,
            "category": category,
            "abstained": not bool(pred.predictions),
            "top1": pred.predictions[0].surface if pred.predictions else None,
            "resolved_context_ids": list(pred.resolved_context_ids),
            "lexical_seed_ids": list(pred.lexical_seed_ids),
            "recall_nodes": int(pred.work.get("recall_nodes", 0)),
            "relation_checks": int(pred.work.get("relation_checks", 0)) + int(pred.work.get("prediction_relation_checks", 0)),
        })
    n = len(rejection_rows)
    abstained = sum(1 for x in rejection_rows if x["abstained"])
    all_unknown = [x for x in rejection_rows if x["category"] == "all_unknown"]
    compositional_unknown = [x for x in rejection_rows if x["category"] != "all_unknown"]
    return {
        "ladder": ladder.to_dict(),
        "unknown_rejection": {
            "cases": n,
            "unknown_rejection_rate": abstained / n if n else 0.0,
            "false_positive_rate": 1.0 - (abstained / n if n else 0.0),
            "all_unknown_rejection_rate": sum(1 for x in all_unknown if x["abstained"]) / len(all_unknown) if all_unknown else 0.0,
            "known_or_mixed_unsupported_rejection_rate": sum(1 for x in compositional_unknown if x["abstained"]) / len(compositional_unknown) if compositional_unknown else 0.0,
            "details": rejection_rows,
            "definition": "Every case is outside compiled predictive support and is expected to abstain; semantic truth is not assumed.",
        },
    }


def run_typed_frame_track(out: Path) -> dict[str, Any]:
    cfg = BehaviourProbeConfig(top_k=3, min_probability=0.0, evidence_confidence=0.95)
    runtime = RuntimeCognitionPolicy(
        max_recall_nodes=96, max_recall_depth=3, max_predictions=3, min_prediction_score=0.0,
        enable_semantic_slot_admission=True,
    )
    frame_policy = FrameInductionPolicy(
        min_target_context_fraction=1.0,
        enable_semantic_slot_typing=True,
        min_filler_coherence_score=0.70,
        min_filler_pair_fraction=1.0,
        slot_admission_min_score=0.70,
        slot_admission_min_support_fillers=2,
        require_typed_slot=True,
    )
    eval_cases = [FrameEvalCase(*row) for row in GAUNTLET_DATA["typed_frame"]["eval"]]
    report = compile_and_benchmark_frame_schemas(
        ScriptedTypedSlotTeacher(), GAUNTLET_DATA["typed_frame"]["train"], eval_cases,
        out / "typed_frame", "v031-typed-frame", "1", probe_cfg=cfg, frame_policy=frame_policy,
        cognition_policy=runtime, runtime_top_k=3, support_documents=[_typed_semantic_support_doc()],
        pack_id="v031-typed-frame", pack_version=31,
    )
    detail = {x.prompt: x for x in report.details}
    stone = detail["Stein ist ein"]
    return {
        "report": report.to_dict(),
        "open_slot_false_positive": stone.open_frame_top1,
        "typed_slot_abstains_on_stone": stone.frame_only_top1 is None,
        "typed_frame_precision": report.frame_applicability_precision,
        "typed_frame_recall": report.frame_applicability_recall,
    }


def _ensure_bank_probe_file(path: Path) -> None:
    rows = [
        {"probe_id": "bank-fin-1", "text": "Die Bank vergibt einen Kredit.", "target": "Bank", "label": "finance"},
        {"probe_id": "bank-fin-2", "text": "Er überweist Geld zur Bank.", "target": "Bank", "label": "finance"},
        {"probe_id": "bank-fin-3", "text": "Das Konto liegt bei der Bank.", "target": "Bank", "label": "finance"},
        {"probe_id": "bank-seat-1", "text": "Die Bank steht im Park.", "target": "Bank", "label": "seat"},
        {"probe_id": "bank-seat-2", "text": "Unter dem Baum steht eine Bank.", "target": "Bank", "label": "seat"},
        {"probe_id": "bank-seat-3", "text": "Am See steht die Bank.", "target": "Bank", "label": "seat"},
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run_ambiguity_track(out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    probe_path = out / "bank_senses.jsonl"
    _ensure_bank_probe_file(probe_path)
    probes = load_probe_dataset([probe_path])
    doc, _, _ = extract_contextual_hidden_evidence(
        ScriptedHiddenBackend(), probes, "v031-hidden", "1",
        HiddenStateConfig(layers=(-1,), top_k=3, min_cosine=.20),
    )
    probe_label = {p.probe_id: p.label for p in probes}
    ctx_label: dict[str, str] = {}
    for e in doc.evidence:
        if e.relation_type == RelationType.CONTEXTUAL_USE.value:
            ctx_label[e.to_id] = probe_label[e.metadata["probe_id"]]
    node_by_surface = {n.surface: n for n in doc.nodes if n.kind == NodeKind.SURFACE.value and n.surface}
    semantic_pairs = [("Geld", "Kredit"), ("Kredit", "Konto"), ("Geld", "Konto"), ("Park", "Baum"), ("Baum", "See"), ("Park", "See")]
    source = SourceRef("v031-static", "1", "embedding_neighbours")
    db_path = out / "ambiguity.sqlite"
    with CanonicalDB(db_path) as db:
        _ingest(db, doc)
        for a, b in semantic_pairs:
            if a not in node_by_surface or b not in node_by_surface:
                raise ValueError(f"ambiguity fixture missing surface {a!r} or {b!r}")
            for x, y in ((a, b), (b, a)):
                db.add_evidence(RelationEvidence(
                    node_by_surface[x].node_id, node_by_surface[y].node_id,
                    RelationType.SEMANTIC_SIMILARITY.value, .90, .90, source,
                    {"fixture": True, "channel": "embedding"},
                ))
        db.commit()
        target = find_text_surface_id(db.db, "Bank")
        derived, report = compile_provisional_senses(db.db, target, SenseCompilePolicy(
            threshold=.62, min_channels=2, min_contexts=2,
            weights=FusionWeights(hidden_similarity=.65, lexical_overlap=.05, semantic_similarity=.25,
                                  substitution=.0, transition=.0, prediction_signature=.05),
        ))
        errors = [x for x in validate_document(derived) if x.severity == "error"]
        if errors:
            raise ValueError("sense compiler validation failed: " + "; ".join(x.message for x in errors))
    cluster_labels = []
    total = correct = 0
    for cluster in report.clusters:
        labels = [ctx_label[x] for x in cluster["contexts"]]
        counts: dict[str, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        total += len(labels)
        correct += max(counts.values()) if counts else 0
        cluster_labels.append({"labels": sorted(labels), "purity": (max(counts.values()) / len(labels) if labels else 0.0)})
    return {
        "contexts_found": report.contexts_found,
        "clusters_created": report.clusters_created,
        "unresolved_contexts": report.unresolved_contexts,
        "cluster_purity": correct / total if total else 0.0,
        "cluster_labels": cluster_labels,
        "gold_labels_used_as_compiler_input": False,
        "compiler_report": report.to_dict(),
    }


def _compile_exception_pack(db: CanonicalDB, path: Path, label: str):
    comp = _pass_through_compression(db.db)
    manifest = compile_base_pack(
        db.db, path, pack_id=f"v031-exception-{label}", pack_version=31,
        compression_run_id=comp.compression_run_id, include_metadata=True,
        relation_policy=CompilePolicy(min_score=0, min_confidence=0, min_sources=1),
    )
    if not manifest.roundtrip_valid:
        raise ValueError("exception gauntlet pack failed roundtrip")
    return manifest, load_base_pack(path)


def _exception_runtime_policy() -> RuntimeCognitionPolicy:
    return RuntimeCognitionPolicy(
        max_recall_nodes=64, max_recall_depth=2, max_predictions=4, min_prediction_score=0.0,
        enable_exact_context=False, enable_suffix_context=False, enable_lexical_seeds=False,
        enable_parent_routing=False, enable_relation_recall=False, enable_transition_prediction=False,
        enable_recalled_context_prediction=False, enable_direct_context_prediction=True,
        enable_frame_schema_matching=True, enable_semantic_slot_admission=True,
        enable_slot_prototype_admission=True, enable_hierarchical_slot_admission=True,
        enable_frame_specificity_override=True, enable_exception_override_frames=True,
        enable_exception_lifecycle_gating=True,
    )


def run_exception_consensus_track(out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    cfg = BehaviourProbeConfig(top_k=2, min_probability=0.0)
    base_doc, _, _ = extract_teacher_behaviour(
        BaselineTeacher(), ["Hund atmet", "Katze atmet", "Adler fliegt", "Falke fliegt"], "v031-base", "1", cfg,
    )
    sup_a, _, _ = extract_teacher_behaviour(ExceptionTeacher(True, "support-A"), ["Adler atmet", "Falke atmet"], "support-A", "1", cfg)
    sup_b, _, _ = extract_teacher_behaviour(ExceptionTeacher(True, "support-B"), ["Adler atmet", "Falke atmet"], "support-B", "1", cfg)
    con_c, _, _ = extract_teacher_behaviour(ExceptionTeacher(False, "conflict-C"), ["Adler atmet", "Falke atmet"], "conflict-C", "1", cfg)
    con_d, _, _ = extract_teacher_behaviour(ExceptionTeacher(False, "conflict-D"), ["Adler atmet", "Falke atmet"], "conflict-D", "1", cfg)

    consensus = ExceptionConsensusPolicy(
        stable_min_sources=2, stable_min_runs=2, stable_consensus=.75, stable_min_support_weight=1.5,
        quarantine_conflicts=2, quarantine_conflict_weight=1.5,
    )
    lifecycle: list[str] = []
    prediction_timeline: list[dict[str, Any]] = []
    db_path = out / "master.sqlite"
    with CanonicalDB(db_path) as db:
        _ingest(db, base_doc)
        _ingest(db, _exception_semantic_doc())
        frame_doc, _ = induce_relation_frames(db.db, FrameInductionPolicy(
            enable_semantic_slot_typing=True, min_filler_coherence_score=.95, min_filler_pair_fraction=1.0,
            require_typed_slot=True, enable_latent_slot_prototype=True, prototype_classify_candidates=True,
            prototype_min_admission_score=.95, prototype_min_support_fillers=2, prototype_membership_confidence=1.0,
        ))
        persist_frame_document(db.db, frame_doc); db.db.commit()
        hierarchy_doc, _ = induce_prototype_hierarchy(db.db, PrototypeHierarchyPolicy(
            max_depth=1, min_cross_leaf_score=.85, score_decay_per_depth=0, min_score_floor=.85,
            min_cross_leaf_pair_fraction=1.0, membership_confidence=.95,
        ))
        persist_frame_document(db.db, hierarchy_doc); db.db.commit()
        transfer_doc, _ = promote_frames_to_higher_concepts(db.db, HierarchicalFrameTransferPolicy(
            min_ancestor_hops=1, max_ancestor_hops=2, min_path_edge_quality=.50,
            hierarchical_admission_min_path_score=.70,
        ))
        persist_frame_document(db.db, transfer_doc); db.db.commit()
        _ingest(db, sup_a); _ingest(db, sup_b)
        override_doc, override_report = induce_exception_overrides(db.db, ExceptionInductionPolicy(
            min_exception_target_score=.70, min_disagreement_margin=.50, min_membership_quality=.80,
            min_prototype_path_quality=.50, max_prototype_hops=2,
        ))
        if override_report.overrides_induced != 1:
            raise AssertionError(f"expected one exception override, got {override_report.overrides_induced}")
        persist_frame_document(db.db, override_doc); db.db.commit()

        def checkpoint(name: str) -> tuple[str, str | None, str | None, str]:
            run = track_exception_run(db.db, override_doc, consensus)
            statuses = sorted({x["status"] for x in run.assessments})
            status = statuses[0] if len(statuses) == 1 else "+".join(statuses)
            lifecycle.append(status)
            _, pack = _compile_exception_pack(db, out / f"{name}.eqbase", name)
            cog = RuntimeCognition(pack, _exception_runtime_policy())
            eule = _top1(cog, "Eule atmet")
            hund = _top1(cog, "Hund atmet")
            prediction_timeline.append({"checkpoint": name, "status": status, "eule_top1": eule, "hund_top1": hund})
            return status, eule, hund, str(pack.manifest.get("full_file_sha256", ""))

        checkpoint("support_run_1")
        _, stable_eule, stable_hund, stable_hash = checkpoint("support_run_2")

        # On the same stable pack, disabling exception frames should restore the broad rule.
        stable_pack = load_base_pack(out / "support_run_2.eqbase")
        base_values = asdict(_exception_runtime_policy())
        base_values["enable_exception_override_frames"] = False
        without_exception = RuntimeCognition(stable_pack, RuntimeCognitionPolicy(**base_values))
        stable_ablation_eule = _top1(without_exception, "Eule atmet")

        _ingest(db, con_c)
        checkpoint("conflict_run_1")
        _ingest(db, con_d)
        checkpoint("conflict_run_2")
        for src in ("conflict-C", "conflict-D"):
            db.db.execute(
                "INSERT OR REPLACE INTO source_policies(source_id,source_version,enabled,trust_weight,note,metadata_json) VALUES(?,?,?,?,?,?)",
                (src, "1", 0, 1.0, "v0.31 gauntlet recovery fixture", "{}"),
            )
        db.db.commit()
        checkpoint("recovery_run_1")
        _, recovered_eule, recovered_hund, recovered_hash = checkpoint("recovery_run_2")
        states = list_exception_states(db.db)

    return {
        "lifecycle": lifecycle,
        "expected_lifecycle": list(GAUNTLET_DATA["exception_consensus"]["expected_lifecycle"]),
        "lifecycle_matches_contract": lifecycle == GAUNTLET_DATA["exception_consensus"]["expected_lifecycle"],
        "prediction_timeline": prediction_timeline,
        "stable_exception_transfer_top1": stable_eule,
        "stable_general_rule_control_top1": stable_hund,
        "stable_ablation_without_exception_top1": stable_ablation_eule,
        "recovered_exception_transfer_top1": recovered_eule,
        "recovered_general_rule_control_top1": recovered_hund,
        "stable_pack_sha256": stable_hash,
        "recovered_pack_sha256": recovered_hash,
        "final_states": states,
    }


def _add_runtime_node(db: CanonicalDB, nid: str, level: int = 1, members: list[str] | None = None) -> None:
    db.upsert_node(IRNode(
        nid, Modality.TEXT.value, NodeKind.COMPOSITE.value if members else NodeKind.SURFACE.value,
        level, nid, members or [], {"fixture": "v031-gauntlet"},
    ))


def run_runtime_routing_track(out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    db_path = out / "routing.sqlite"
    pack_path = out / "routing.eqbase"
    parent_count = int(GAUNTLET_DATA["runtime_routing"]["parents"])
    with CanonicalDB(db_path) as c:
        for nid in ("a", "b", "c"):
            _add_runtime_node(c, nid)
        src = SourceRef("v031-runtime-routing", "1", "runtime_benchmark_fixture")
        for i in range(parent_count):
            left, right, parent = f"left-{i:03d}", f"right-{i:03d}", f"parent-{i:03d}"
            _add_runtime_node(c, left); _add_runtime_node(c, right); _add_runtime_node(c, parent, 2, [left, "a", "b", "c", right])
            c.add_evidence(RelationEvidence(
                "a", right, RelationType.PREDICTION.value, 0.5 + (i % 10) / 100.0, 0.9, src, {"fixture_index": i},
            ))
        c.commit()
        discovery = MacroDiscoveryPolicy(
            min_pattern_size=2, max_pattern_size=3, min_occurrences=3, min_parent_structures=3,
            definition_overhead=1.0, min_net_savings=1.0, min_compression_ratio=0.0, prune_dominated=True,
        )
        promotion = MacroPromotionPolicy(
            min_realized_occurrences=3, min_realized_parent_structures=3, definition_overhead=1.0,
            min_realized_net_savings=1.0, min_realized_compression_ratio=0.0, max_promotions=8,
        )
        compression = compile_runtime_compression(c.db, discovery, promotion, persist=True, materialize=True)
        abc = next(x for x in compression.promoted if tuple(x.members) == ("a", "b", "c"))
        manifest = compile_base_pack(
            c.db, pack_path, pack_id="v031-routing", pack_version=31,
            compression_run_id=compression.compression_run_id, include_metadata=False,
            persist=False, verify_roundtrip=True,
        )
        if not manifest.roundtrip_valid:
            raise AssertionError("runtime routing pack roundtrip failed")
    report = benchmark_runtime_pack(pack_path, sample_nodes=128, sample_macros=8, timing_repeat=5, max_expanded_symbols=1_000_000)
    route = next(x for x in report.routing if x["macro_node_id"] == abc.macro_node_id)
    return {
        "pack_manifest": manifest.to_dict(),
        "benchmark": report.to_dict(),
        "target_macro": abc.macro_node_id,
        "target_macro_members": list(abc.members),
        "target_macro_route": route,
    }


@dataclass
class BaselineGauntletReport:
    format: str
    version: str
    dataset_sha256: str
    elapsed_ms: float
    tracks: dict[str, Any]
    headline_metrics: dict[str, Any]
    exposed_failures: list[str]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_baseline_gauntlet(out_dir: str | Path) -> BaselineGauntletReport:
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gauntlet_cases.json").write_text(
        json.dumps(GAUNTLET_DATA, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    start = time.perf_counter()
    tracks = {
        "generalization_and_rejection": run_generalization_and_rejection(out / "01_generalization_and_rejection"),
        "typed_frame_admission": run_typed_frame_track(out / "02_typed_frame_admission"),
        "ambiguity_separation": run_ambiguity_track(out / "03_ambiguity_separation"),
        "exception_consensus": run_exception_consensus_track(out / "04_exception_consensus"),
        "runtime_routing": run_runtime_routing_track(out / "05_runtime_routing"),
    }
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    ladder_levels = tracks["generalization_and_rejection"]["ladder"]["level_reports"]
    level_top1 = {int(x["level"]): float(x["top1_retention"]) for x in ladder_levels}
    rejection = tracks["generalization_and_rejection"]["unknown_rejection"]
    typed = tracks["typed_frame_admission"]
    ambiguity = tracks["ambiguity_separation"]
    consensus = tracks["exception_consensus"]
    route = tracks["runtime_routing"]["target_macro_route"]
    headline = {
        "exact_replay_top1": level_top1.get(0, 0.0),
        "known_suffix_transfer_top1": level_top1.get(1, 0.0),
        "known_parts_recombination_top1": level_top1.get(2, 0.0),
        "novel_order_known_lexicon_top1": level_top1.get(3, 0.0),
        "novel_entity_known_frame_top1": level_top1.get(4, 0.0),
        "heldout_relation_family_top1": level_top1.get(5, 0.0),
        "unknown_rejection_rate": rejection["unknown_rejection_rate"],
        "known_or_mixed_unsupported_rejection_rate": rejection["known_or_mixed_unsupported_rejection_rate"],
        "typed_frame_precision": typed["typed_frame_precision"],
        "typed_frame_recall": typed["typed_frame_recall"],
        "ambiguity_cluster_purity": ambiguity["cluster_purity"],
        "exception_lifecycle_contract": consensus["lifecycle_matches_contract"],
        "stable_exception_transfer_top1": consensus["stable_exception_transfer_top1"],
        "runtime_macro_route_work_reduction": route["work_reduction"],
    }
    failures: list[str] = []
    if headline["heldout_relation_family_top1"] < 1.0:
        failures.append("Held-out relation-family transfer fails in the frozen L5 case.")
    if headline["known_or_mixed_unsupported_rejection_rate"] < 1.0:
        failures.append("Lexical recall produces false positives for unsupported recombinations of known tokens.")
    if tracks["typed_frame_admission"]["open_slot_false_positive"] is not None:
        failures.append("Untyped open-frame matching overgeneralizes to the Stein counterexample; typed admission is required to suppress it.")

    report = BaselineGauntletReport(
        format=GAUNTLET_FORMAT,
        version=GAUNTLET_VERSION,
        dataset_sha256=gauntlet_dataset_sha256(),
        elapsed_ms=elapsed_ms,
        tracks=tracks,
        headline_metrics=headline,
        exposed_failures=failures,
        notes=[
            "v0.31 adds no cognitive mechanism. It freezes an adversarial deterministic benchmark over the v0.30/v0.29 cognition semantics.",
            "No composite overall score is emitted; each failure family remains visible independently.",
            "Timing fields are observational and must not be used as strict regression assertions across machines.",
            "The ambiguity track uses gold labels only for evaluation purity, never as compiler input.",
            "Unsupported-query rejection means lack of compiled predictive support, not a claim that the natural-language prompt is false or meaningless.",
        ],
    )
    (out / "baseline-gauntlet-report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the frozen Equinox Forge v0.31 Baseline Gauntlet")
    p.add_argument("--out", default="out/v0_31_baseline_gauntlet", help="output directory")
    p.add_argument("--json", action="store_true", help="print full JSON report instead of headline summary")
    args = p.parse_args(argv)
    report = run_baseline_gauntlet(args.out)
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(json.dumps({
            "format": report.format,
            "version": report.version,
            "dataset_sha256": report.dataset_sha256,
            "headline_metrics": report.headline_metrics,
            "exposed_failures": report.exposed_failures,
            "report": str(Path(args.out) / "baseline-gauntlet-report.json"),
        }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
