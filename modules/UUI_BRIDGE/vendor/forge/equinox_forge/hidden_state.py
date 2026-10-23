from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Protocol
import json
import math
import numpy as np

from .canonical import make_text_surface, make_composite, normalize_surface
from .ir import ForgeDocument, IRNode, RelationEvidence, RelationType, SourceRef, NodeKind, Modality
from .language_evidence import tokenize_text, LanguageEvidenceConfig

HIDDEN_STATE_FORMAT = "equinox-forge-contextual-hidden-v1"
PROBE_DATASET_FORMAT = "equinox-forge-probe-dataset-v1"


@dataclass(frozen=True)
class ContextProbeItem:
    probe_id: str
    text: str
    target: str
    occurrence: int = 1
    label: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class HiddenStateSample:
    layer: int
    vector: list[float]
    token_count: int
    metadata: dict[str, Any] | None = None


class ContextualHiddenBackend(Protocol):
    backend_id: str

    def contextual_state(
        self,
        text: str,
        target_start: int,
        target_end: int,
        layers: list[int],
        pooling: str,
    ) -> list[HiddenStateSample]: ...

    def manifest(self) -> dict[str, Any]: ...


@dataclass
class HiddenStateConfig:
    layers: tuple[int, ...] = (-1,)
    pooling: str = "mean"
    top_k: int = 6
    min_cosine: float = 0.25
    evidence_confidence: float = 0.85
    max_context_words: int = 96
    include_cross_surface: bool = False
    keep_vectors_in_report: bool = False

    def validate(self) -> None:
        if not self.layers:
            raise ValueError("at least one layer is required")
        if self.pooling not in {"mean", "first", "last"}:
            raise ValueError("pooling must be mean, first or last")
        if self.top_k <= 0:
            raise ValueError("top_k must be > 0")
        if not -1.0 <= self.min_cosine <= 1.0:
            raise ValueError("min_cosine must be in [-1, 1]")
        if not 0.0 <= self.evidence_confidence <= 1.0:
            raise ValueError("evidence_confidence must be in [0, 1]")
        if self.max_context_words <= 0:
            raise ValueError("max_context_words must be > 0")


@dataclass
class HiddenProbeRecord:
    probe_id: str
    target: str
    canonical_target_id: str
    context_id: str
    label: str | None
    layer: int
    dimension: int
    token_count: int
    vector_norm: float
    vector: list[float] | None = None


@dataclass
class HiddenStateReport:
    format: str
    source_id: str
    source_version: str
    backend_id: str
    probes_requested: int
    probes_encoded: int
    surfaces: int
    context_nodes: int
    hidden_samples: int
    contextual_use_evidence: int
    contextual_similarity_evidence: int
    skipped_missing_target: int
    skipped_backend_empty: int
    layers: list[int]
    pooling: str
    top_k: int
    min_cosine: float
    labelled_pairs_same: int = 0
    labelled_pairs_different: int = 0
    mean_cosine_same_label: float | None = None
    mean_cosine_different_label: float | None = None
    nearest_neighbour_label_accuracy: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _find_occurrence(text: str, target: str, occurrence: int) -> tuple[int, int] | None:
    if occurrence <= 0:
        raise ValueError("occurrence is 1-based and must be > 0")
    start = 0
    found = -1
    for _ in range(occurrence):
        found = text.find(target, start)
        if found < 0:
            return None
        start = found + max(1, len(target))
    return found, found + len(target)


def _context_tokens(text: str, cfg: HiddenStateConfig) -> list[str]:
    lc = LanguageEvidenceConfig(include_punctuation=False)
    sentences = tokenize_text(normalize_surface(text), lc)
    tokens = [t for sent in sentences for t in sent]
    if len(tokens) > cfg.max_context_words:
        tokens = tokens[-cfg.max_context_words:]
    return tokens


def _context_node(item: ContextProbeItem, start: int, end: int, cfg: HiddenStateConfig) -> tuple[list[IRNode], IRNode, IRNode]:
    target_q, target_node = make_text_surface(normalize_surface(item.target))
    nodes: dict[str, IRNode] = {n.node_id: n for n in [*target_q, target_node]}
    surfaces: list[IRNode] = []
    for token in _context_tokens(item.text, cfg):
        qs, s = make_text_surface(token)
        for n in [*qs, s]:
            nodes[n.node_id] = n
        surfaces.append(s)
    if not surfaces:
        surfaces = [target_node]
    marked = normalize_surface(item.text[:start] + "⟦" + item.text[start:end] + "⟧" + item.text[end:])
    context = make_composite(
        Modality.TEXT.value,
        surfaces,
        kind=NodeKind.TEMPLATE.value,
        label=marked,
    )
    nodes[context.node_id] = context
    return list(nodes.values()), target_node, context


def _unit(vec: Iterable[float]) -> np.ndarray:
    x = np.asarray(list(vec), dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("hidden vector must be a finite non-empty 1D vector")
    n = float(np.linalg.norm(x))
    if n <= 0.0:
        raise ValueError("hidden vector norm must be > 0")
    return x / n


def _pair_cos(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"hidden vector dimension mismatch: {a.shape} != {b.shape}")
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def _score_from_cos(cosine: float) -> float:
    # Relation scores are [0,1]. Preserve raw cosine separately in metadata.
    return float(max(0.0, min(1.0, cosine)))


def load_probe_dataset(paths: Iterable[str | Path]) -> list[ContextProbeItem]:
    out: list[ContextProbeItem] = []
    seen: set[str] = set()
    for path in paths:
        p = Path(path)
        for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            raw = json.loads(line)
            if isinstance(raw, dict) and raw.get("format") == PROBE_DATASET_FORMAT:
                continue
            probe_id = str(raw.get("id") or f"{p.name}:{lineno}")
            if probe_id in seen:
                raise ValueError(f"duplicate probe id: {probe_id}")
            seen.add(probe_id)
            out.append(ContextProbeItem(
                probe_id=probe_id,
                text=str(raw["text"]),
                target=str(raw["target"]),
                occurrence=int(raw.get("occurrence", 1)),
                label=None if raw.get("label") is None else str(raw.get("label")),
                metadata=dict(raw.get("metadata") or {}),
            ))
    return out


def _label_metrics(records: list[tuple[ContextProbeItem, int, np.ndarray]]) -> dict[str, Any]:
    same: list[float] = []
    diff: list[float] = []
    nn_correct = 0
    nn_total = 0
    # only compare identical target surface and identical layer
    for i, (ai, al, av) in enumerate(records):
        if ai.label is None:
            continue
        candidates: list[tuple[float, ContextProbeItem]] = []
        for j, (bi, bl, bv) in enumerate(records):
            if i == j or al != bl or ai.target != bi.target or bi.label is None:
                continue
            c = _pair_cos(av, bv)
            (same if ai.label == bi.label else diff).append(c)
            candidates.append((c, bi))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            nn_total += 1
            nn_correct += int(candidates[0][1].label == ai.label)
    # pairs were observed from both directions; mean is unaffected.
    return {
        "labelled_pairs_same": len(same) // 2,
        "labelled_pairs_different": len(diff) // 2,
        "mean_cosine_same_label": None if not same else float(np.mean(same)),
        "mean_cosine_different_label": None if not diff else float(np.mean(diff)),
        "nearest_neighbour_label_accuracy": None if not nn_total else float(nn_correct / nn_total),
    }


def extract_contextual_hidden_evidence(
    backend: ContextualHiddenBackend,
    probes: Iterable[ContextProbeItem],
    source_id: str,
    source_version: str,
    cfg: HiddenStateConfig | None = None,
) -> tuple[ForgeDocument, HiddenStateReport, list[HiddenProbeRecord]]:
    cfg = cfg or HiddenStateConfig()
    cfg.validate()
    probe_list = list(probes)
    src_use = SourceRef(source_id, source_version, "contextual_hidden_use")
    src_sim = SourceRef(source_id, source_version, "contextual_hidden_similarity")
    node_map: dict[str, IRNode] = {}
    evidence: list[RelationEvidence] = []
    records: list[HiddenProbeRecord] = []
    vectors: list[tuple[ContextProbeItem, IRNode, IRNode, int, np.ndarray, HiddenStateSample]] = []
    skipped_missing = skipped_empty = 0

    for item in probe_list:
        text = normalize_surface(item.text)
        target = normalize_surface(item.target)
        span = _find_occurrence(text, target, item.occurrence)
        if span is None:
            skipped_missing += 1
            continue
        start, end = span
        nodes, target_node, context = _context_node(item, start, end, cfg)
        for n in nodes:
            node_map[n.node_id] = n
        samples = backend.contextual_state(text, start, end, list(cfg.layers), cfg.pooling)
        if not samples:
            skipped_empty += 1
            continue
        for sample in samples:
            vec = _unit(sample.vector)
            vectors.append((item, target_node, context, sample.layer, vec, sample))
            evidence.append(RelationEvidence(
                from_id=target_node.node_id,
                to_id=context.node_id,
                relation_type=RelationType.CONTEXTUAL_USE.value,
                score=1.0,
                confidence=float(cfg.evidence_confidence),
                source=src_use,
                metadata={
                    "evidence_kind": "contextual_hidden_use",
                    "backend_id": backend.backend_id,
                    "probe_id": item.probe_id,
                    "target": target,
                    "target_start": start,
                    "target_end": end,
                    "layer": int(sample.layer),
                    "pooling": cfg.pooling,
                    "dimension": int(vec.size),
                    "token_count": int(sample.token_count),
                    "backend_metadata": sample.metadata or {},
                },
            ))
            records.append(HiddenProbeRecord(
                probe_id=item.probe_id,
                target=target,
                canonical_target_id=target_node.node_id,
                context_id=context.node_id,
                label=item.label,
                layer=int(sample.layer),
                dimension=int(vec.size),
                token_count=int(sample.token_count),
                vector_norm=float(np.linalg.norm(np.asarray(sample.vector, dtype=np.float64))),
                vector=vec.astype(float).tolist() if cfg.keep_vectors_in_report else None,
            ))

    # Sparse contextual similarity graph. By default compare only identical
    # surfaces; cross-surface semantic similarity remains a distinct future channel.
    grouped: dict[tuple[str, int], list[tuple[ContextProbeItem, IRNode, IRNode, int, np.ndarray, HiddenStateSample]]] = {}
    for row in vectors:
        item, target_node, _, layer, _, _ = row
        key = ("*" if cfg.include_cross_surface else target_node.node_id, layer)
        grouped.setdefault(key, []).append(row)

    sim_count = 0
    seen_pairs: set[tuple[str, str, int]] = set()
    for (_, layer), rows in grouped.items():
        for i, a in enumerate(rows):
            ai, atarget, actx, _, avec, asample = a
            scored: list[tuple[float, Any]] = []
            for j, b in enumerate(rows):
                if i == j:
                    continue
                bi, btarget, bctx, _, bvec, bsample = b
                cos = _pair_cos(avec, bvec)
                if cos >= cfg.min_cosine:
                    scored.append((cos, b))
            scored.sort(key=lambda x: (-x[0], x[1][2].node_id))
            for cos, b in scored[:cfg.top_k]:
                bi, btarget, bctx, _, bvec, bsample = b
                lo, hi = sorted((actx.node_id, bctx.node_id))
                pair_key = (lo, hi, layer)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                evidence.append(RelationEvidence(
                    from_id=lo,
                    to_id=hi,
                    relation_type=RelationType.CONTEXTUAL_SIMILARITY.value,
                    score=_score_from_cos(cos),
                    confidence=float(cfg.evidence_confidence),
                    source=src_sim,
                    metadata={
                        "evidence_kind": "contextual_hidden_similarity",
                        "backend_id": backend.backend_id,
                        "layer": int(layer),
                        "pooling": cfg.pooling,
                        "raw_cosine": float(cos),
                        "probe_ids": sorted([ai.probe_id, bi.probe_id]),
                        "targets": sorted([ai.target, bi.target]),
                        "dimension": int(avec.size),
                    },
                ))
                sim_count += 1

    metric_rows = [(item, layer, vec) for item, _, _, layer, vec, _ in vectors]
    metrics = _label_metrics(metric_rows)
    nodes = sorted(node_map.values(), key=lambda n: (n.level, n.node_id))
    report = HiddenStateReport(
        format=HIDDEN_STATE_FORMAT,
        source_id=source_id,
        source_version=source_version,
        backend_id=backend.backend_id,
        probes_requested=len(probe_list),
        probes_encoded=len({r.probe_id for r in records}),
        surfaces=len({r.canonical_target_id for r in records}),
        context_nodes=len({r.context_id for r in records}),
        hidden_samples=len(records),
        contextual_use_evidence=len(records),
        contextual_similarity_evidence=sim_count,
        skipped_missing_target=skipped_missing,
        skipped_backend_empty=skipped_empty,
        layers=list(cfg.layers),
        pooling=cfg.pooling,
        top_k=cfg.top_k,
        min_cosine=cfg.min_cosine,
        **metrics,
    )
    return ForgeDocument(nodes=nodes, evidence=evidence), report, records


class TransformersHiddenStateBackend:
    """Optional Hugging-Face runtime backend.

    Requires the `hf-runtime` extra. It uses tokenizer offset mappings so the
    teacher's subword boundaries never become Equinox canonical boundaries.
    """

    backend_id = "transformers-hidden-state"

    def __init__(self, model_dir: str | Path, device: str = "cpu", dtype: str = "auto", trust_remote_code: bool = False):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Transformers hidden-state backend requires torch+transformers+tokenizers") from exc
        self.torch = torch
        self.model_dir = str(model_dir)
        self.device = device
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True, trust_remote_code=trust_remote_code, local_files_only=True)
        if not getattr(self.tokenizer, "is_fast", False):
            raise RuntimeError("a fast tokenizer with offset_mapping support is required")
        kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if dtype != "auto":
            name = dtype.replace("torch.", "")
            if not hasattr(torch, name):
                raise ValueError(f"unknown torch dtype: {dtype}")
            kwargs["torch_dtype"] = getattr(torch, name)
        self.model = AutoModel.from_pretrained(self.model_dir, local_files_only=True, **kwargs).to(device)
        self.model.eval()

    def manifest(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "model_dir": self.model_dir,
            "device": self.device,
            "dtype": self.dtype,
            "model_class": self.model.__class__.__name__,
            "tokenizer_class": self.tokenizer.__class__.__name__,
        }

    def contextual_state(self, text: str, target_start: int, target_end: int, layers: list[int], pooling: str) -> list[HiddenStateSample]:
        enc = self.tokenizer(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        model_inputs = {k: v.to(self.device) for k, v in enc.items()}
        with self.torch.no_grad():
            out = self.model(**model_inputs, output_hidden_states=True, return_dict=True)
        hidden = out.hidden_states
        token_idxs = [i for i, (a, b) in enumerate(offsets) if b > a and a < target_end and b > target_start]
        if not token_idxs:
            return []
        result: list[HiddenStateSample] = []
        for requested in layers:
            idx = requested if requested >= 0 else len(hidden) + requested
            if idx < 0 or idx >= len(hidden):
                raise IndexError(f"hidden layer {requested} resolves to {idx}, available=0..{len(hidden)-1}")
            states = hidden[idx][0, token_idxs, :].detach().float().cpu()
            if pooling == "mean": vec = states.mean(dim=0)
            elif pooling == "first": vec = states[0]
            elif pooling == "last": vec = states[-1]
            else: raise ValueError(pooling)
            result.append(HiddenStateSample(
                layer=requested,
                vector=vec.tolist(),
                token_count=len(token_idxs),
                metadata={"resolved_layer": idx, "token_indices": token_idxs},
            ))
        return result
