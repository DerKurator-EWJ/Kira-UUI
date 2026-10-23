from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .canonical import make_text_surface
from .hf_adapter import HFAdapterSpec, HFModelManifest, ExtractionPlan, build_extraction_plan, adapter_hash
from .ir import ForgeDocument, RelationEvidence, RelationType, SourceRef

try:
    from safetensors import safe_open
except Exception as exc:  # pragma: no cover - surfaced with a better message at runtime
    safe_open = None
    _SAFETENSORS_IMPORT_ERROR = exc
else:
    _SAFETENSORS_IMPORT_ERROR = None


EXTRACTOR_FORMAT = "equinox-forge-text-teacher-extractor-v1"


def _gpt2_byte_decoder() -> dict[str, int]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_GPT2_BYTE_DECODER = _gpt2_byte_decoder()



@dataclass(frozen=True)
class TokenEntry:
    token_id: int
    raw_token: str
    surface: str
    special: bool = False


@dataclass
class TextExtractionConfig:
    max_tokens: int = 4096
    start_token_id: int = 0
    top_k: int = 8
    min_similarity: float = 0.20
    batch_size: int = 256
    max_surface_chars: int = 64
    include_leading_space: bool = False
    include_punctuation_only: bool = False
    include_numeric_only: bool = True
    normalize_embeddings: bool = True
    evidence_confidence: float = 0.55

    def validate(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if self.start_token_id < 0:
            raise ValueError("start_token_id must be >= 0")
        if self.top_k <= 0:
            raise ValueError("top_k must be > 0")
        if not -1.0 <= self.min_similarity <= 1.0:
            raise ValueError("min_similarity must be in [-1, 1]")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.max_surface_chars <= 0:
            raise ValueError("max_surface_chars must be > 0")
        if not 0.0 <= self.evidence_confidence <= 1.0:
            raise ValueError("evidence_confidence must be in [0, 1]")


@dataclass
class TextExtractionReport:
    format: str
    source_id: str
    source_version: str
    adapter_id: str
    adapter_hash: str
    embedding_tensor: str
    embedding_shape: list[int]
    tokenizer_entries: int
    selected_teacher_tokens: int
    canonical_surfaces: int
    q_nodes: int
    surface_nodes: int
    evidence_records: int
    skipped_special: int
    skipped_invalid_surface: int
    duplicate_surface_tokens: int
    top_k: int
    min_similarity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_special_ids(root: Path) -> set[int]:
    special: set[int] = set()
    tok_path = root / "tokenizer.json"
    if tok_path.exists():
        raw = _read_json(tok_path)
        for item in raw.get("added_tokens", []) or []:
            try:
                if item.get("special"):
                    special.add(int(item["id"]))
            except Exception:
                pass
    cfg_path = root / "tokenizer_config.json"
    if cfg_path.exists():
        raw = _read_json(cfg_path)
        for k, v in (raw.get("added_tokens_decoder", {}) or {}).items():
            if isinstance(v, dict) and v.get("special"):
                try:
                    special.add(int(k))
                except Exception:
                    pass
    return special


def _vocab_from_tokenizer_json(path: Path) -> dict[int, str]:
    raw = _read_json(path)
    model = raw.get("model", {}) or {}
    vocab = model.get("vocab")
    out: dict[int, str] = {}
    if isinstance(vocab, dict):
        for token, token_id in vocab.items():
            try:
                out[int(token_id)] = str(token)
            except Exception:
                continue
    elif isinstance(vocab, list):
        # SentencePiece/Unigram tokenizer.json commonly stores [[piece, score], ...].
        for token_id, item in enumerate(vocab):
            if isinstance(item, (list, tuple)) and item:
                out[token_id] = str(item[0])
            elif isinstance(item, str):
                out[token_id] = item
    # Added tokens may lie outside the model vocab but still have stable IDs.
    for item in raw.get("added_tokens", []) or []:
        try:
            out.setdefault(int(item["id"]), str(item["content"]))
        except Exception:
            pass
    return out


def _vocab_from_vocab_json(path: Path) -> dict[int, str]:
    raw = _read_json(path)
    if not isinstance(raw, dict):
        return {}
    out: dict[int, str] = {}
    for token, token_id in raw.items():
        try:
            out[int(token_id)] = str(token)
        except Exception:
            continue
    return out


def load_local_tokenizer_vocab(model_dir: str | Path) -> list[tuple[int, str, bool]]:
    """Load the static local tokenizer vocabulary without transformers/tokenizers.

    Supports the common JSON forms used by BPE/WordPiece/Unigram tokenizers.
    Binary tokenizer.model/spiece.model alone are intentionally not parsed in v0.3.
    """
    root = Path(model_dir)
    vocab: dict[int, str] = {}
    if (root / "tokenizer.json").exists():
        vocab.update(_vocab_from_tokenizer_json(root / "tokenizer.json"))
    if not vocab and (root / "vocab.json").exists():
        vocab.update(_vocab_from_vocab_json(root / "vocab.json"))
    if not vocab:
        raise ValueError(
            "no JSON tokenizer vocabulary found; v0.3 supports tokenizer.json or vocab.json. "
            "A model with only tokenizer.model/spiece.model needs a future tokenizer backend."
        )
    special_ids = _collect_special_ids(root)
    return [(i, vocab[i], i in special_ids) for i in sorted(vocab)]


def decode_token_surface(raw_token: str, include_leading_space: bool = False) -> str:
    """Best-effort static rendering for common local HF tokenizer vocabularies.

    GPT-2/ByteLevel BPE's reversible byte alphabet is decoded directly. SentencePiece
    and WordPiece boundary markers are normalized. This is still intentionally not a
    full tokenizer pipeline: model-specific normalizers/post-processors are not run.
    """
    s = raw_token
    # GPT-2 / ByteLevel BPE maps raw bytes into a reversible Unicode alphabet.
    if s and all(ch in _GPT2_BYTE_DECODER for ch in s):
        try:
            raw = bytes(_GPT2_BYTE_DECODER[ch] for ch in s)
            s = raw.decode("utf-8")
        except UnicodeDecodeError:
            # A token can represent only a fragment of a multi-byte codepoint; preserve
            # the raw form rather than inventing replacement characters.
            s = raw_token
    # SentencePiece boundary marker. ByteLevel's Ġ usually disappeared above into a space.
    leading_boundary = s.startswith("▁") or s.startswith("Ġ") or s.startswith(" ")
    if s.startswith("▁") or s.startswith("Ġ"):
        s = s[1:]
        if include_leading_space:
            s = " " + s
    elif s.startswith(" ") and not include_leading_space:
        s = s[1:]
    # WordPiece continuation marker is metadata, not a literal surface character.
    if s.startswith("##"):
        s = s[2:]
    return s


def _valid_surface(surface: str, cfg: TextExtractionConfig) -> bool:
    if not surface or len(surface) > cfg.max_surface_chars:
        return False
    if any(ord(ch) < 32 and ch not in {"\t", "\n"} for ch in surface):
        return False
    if not any(ch.isprintable() or ch in {"\t", "\n"} for ch in surface):
        return False
    stripped = surface.strip()
    if not stripped:
        return False
    if not cfg.include_punctuation_only and not any(ch.isalnum() for ch in stripped):
        return False
    if not cfg.include_numeric_only and stripped.replace(".", "", 1).isdigit():
        return False
    return True


def select_token_entries(model_dir: str | Path, cfg: TextExtractionConfig) -> tuple[list[TokenEntry], dict[str, int]]:
    cfg.validate()
    vocab = load_local_tokenizer_vocab(model_dir)
    entries: list[TokenEntry] = []
    skipped_special = 0
    skipped_invalid = 0
    start = cfg.start_token_id
    for token_id, raw_token, special in vocab:
        if token_id < start:
            continue
        if special:
            skipped_special += 1
            continue
        if raw_token.startswith("<0x") and raw_token.endswith(">"):
            skipped_invalid += 1
            continue
        surface = decode_token_surface(raw_token, cfg.include_leading_space)
        if not _valid_surface(surface, cfg):
            skipped_invalid += 1
            continue
        entries.append(TokenEntry(token_id, raw_token, surface, special=False))
        if len(entries) >= cfg.max_tokens:
            break
    return entries, {
        "tokenizer_entries": len(vocab),
        "skipped_special": skipped_special,
        "skipped_invalid_surface": skipped_invalid,
    }


def _resolved_embedding_tensor(plan: ExtractionPlan) -> str:
    for role in plan.resolved_roles:
        if role.role == "input_embeddings" and role.matches:
            return role.matches[0]
    raise ValueError("extraction plan has no resolved input_embeddings tensor")


def _tensor_source(manifest: HFModelManifest, tensor_name: str) -> Path:
    shard = manifest.tensor_sources.get(tensor_name)
    if not shard:
        raise KeyError(f"no safetensors source registered for {tensor_name!r}")
    p = Path(manifest.model_dir) / shard
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def load_embedding_rows(manifest: HFModelManifest, tensor_name: str, token_ids: Iterable[int]) -> tuple[np.ndarray, tuple[int, ...]]:
    """Load only the bounded row interval covering selected IDs from Safetensors.

    This deliberately avoids materializing the full embedding matrix for large teachers.
    The selected IDs should normally form a bounded range from the vocabulary sampler.
    """
    if safe_open is None:
        raise RuntimeError(f"safetensors is required for v0.3 extraction: {_SAFETENSORS_IMPORT_ERROR}")
    ids = np.asarray(list(token_ids), dtype=np.int64)
    if ids.size == 0:
        raise ValueError("no token ids selected")
    shard = _tensor_source(manifest, tensor_name)
    with safe_open(str(shard), framework="np", device="cpu") as f:
        sl = f.get_slice(tensor_name)
        shape = tuple(int(x) for x in sl.get_shape())
        if len(shape) != 2:
            raise ValueError(f"embedding tensor must be rank-2, got {shape}")
        if int(ids.min()) < 0 or int(ids.max()) >= shape[0]:
            raise IndexError(f"token id outside embedding rows: min={ids.min()} max={ids.max()} rows={shape[0]}")
        lo, hi = int(ids.min()), int(ids.max()) + 1
        block = np.asarray(sl[lo:hi], dtype=np.float32)
    rows = block[ids - lo]
    return rows, shape


def _surface_groups(entries: list[TokenEntry], vectors: np.ndarray) -> tuple[list[str], np.ndarray, dict[str, list[TokenEntry]]]:
    groups: dict[str, list[int]] = {}
    meta: dict[str, list[TokenEntry]] = {}
    for idx, e in enumerate(entries):
        groups.setdefault(e.surface, []).append(idx)
        meta.setdefault(e.surface, []).append(e)
    surfaces = sorted(groups, key=lambda s: min(x.token_id for x in meta[s]))
    out = np.empty((len(surfaces), vectors.shape[1]), dtype=np.float32)
    for i, surface in enumerate(surfaces):
        out[i] = vectors[groups[surface]].mean(axis=0)
    return surfaces, out, meta


def sparse_cosine_neighbors(vectors: np.ndarray, top_k: int, min_similarity: float, batch_size: int = 256) -> list[list[tuple[int, float]]]:
    if vectors.ndim != 2:
        raise ValueError("vectors must be rank-2")
    n = vectors.shape[0]
    if n <= 1:
        return [[] for _ in range(n)]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = vectors / norms
    k = min(top_k, n - 1)
    result: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        sims = unit[start:end] @ unit.T
        for local, row in enumerate(sims):
            i = start + local
            row[i] = -np.inf
            # argpartition avoids a full O(n log n) sort for every row.
            cand = np.argpartition(row, -k)[-k:]
            ordered = cand[np.argsort(row[cand])[::-1]]
            result[i] = [(int(j), float(row[j])) for j in ordered if float(row[j]) >= min_similarity]
    return result


def extract_text_teacher(
    manifest: HFModelManifest,
    spec: HFAdapterSpec,
    source_id: str,
    source_version: str,
    cfg: TextExtractionConfig | None = None,
) -> tuple[ForgeDocument, TextExtractionReport, dict[str, list[TokenEntry]]]:
    cfg = cfg or TextExtractionConfig()
    cfg.validate()
    plan = build_extraction_plan(manifest, spec)
    if not plan.valid:
        raise ValueError("invalid extraction plan: " + "; ".join(plan.warnings))
    if "token_embedding_neighbors" not in plan.enabled_channels:
        raise ValueError("adapter/model does not enable token_embedding_neighbors")
    tensor_name = _resolved_embedding_tensor(plan)
    entries, selection_stats = select_token_entries(manifest.model_dir, cfg)
    if not entries:
        raise ValueError("no valid teacher tokens selected")
    vectors, embedding_shape = load_embedding_rows(manifest, tensor_name, [e.token_id for e in entries])
    surfaces, surface_vectors, surface_meta = _surface_groups(entries, vectors)

    doc = ForgeDocument()
    surface_nodes = []
    q_by_id = {}
    node_by_surface = {}
    for surface in surfaces:
        qs, node = make_text_surface(surface)
        # Canonical node metadata must be source-independent. Teacher token IDs/raw
        # spellings are stored as source aliases by the ingest path, not on this node.
        node.metadata.update({"surface_role": "text_surface"})
        for q in qs:
            q_by_id.setdefault(q.node_id, q)
        surface_nodes.append(node)
        node_by_surface[surface] = node
    doc.nodes.extend(q_by_id.values())
    doc.nodes.extend(surface_nodes)

    neighbors = sparse_cosine_neighbors(surface_vectors, cfg.top_k, cfg.min_similarity, cfg.batch_size)
    source = SourceRef(source_id=source_id, source_version=source_version, extraction="token_embedding_neighbors")
    seen_edges: set[tuple[str, str]] = set()
    for i, links in enumerate(neighbors):
        a = node_by_surface[surfaces[i]]
        for j, sim in links:
            b = node_by_surface[surfaces[j]]
            if a.node_id == b.node_id:
                continue
            key = (a.node_id, b.node_id)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            # Cosine can be [-1,1]; this channel only emits similarities above the configured threshold.
            score = max(0.0, min(1.0, sim))
            doc.evidence.append(RelationEvidence(
                from_id=a.node_id,
                to_id=b.node_id,
                relation_type=RelationType.SEMANTIC_SIMILARITY.value,
                score=score,
                # Confidence is deliberately NOT derived from cosine magnitude. Cosine is
                # the relation score; confidence expresses trust in this extraction channel.
                confidence=cfg.evidence_confidence,
                source=source,
                metadata={
                    "teacher_signal": "input_embedding_cosine",
                    "cosine": sim,
                    "adapter_id": spec.adapter_id,
                    "adapter_hash": adapter_hash(spec),
                    "embedding_tensor": tensor_name,
                },
            ))

    report = TextExtractionReport(
        format=EXTRACTOR_FORMAT,
        source_id=source_id,
        source_version=source_version,
        adapter_id=spec.adapter_id,
        adapter_hash=adapter_hash(spec),
        embedding_tensor=tensor_name,
        embedding_shape=list(embedding_shape),
        tokenizer_entries=selection_stats["tokenizer_entries"],
        selected_teacher_tokens=len(entries),
        canonical_surfaces=len(surfaces),
        q_nodes=len(q_by_id),
        surface_nodes=len(surface_nodes),
        evidence_records=len(doc.evidence),
        skipped_special=selection_stats["skipped_special"],
        skipped_invalid_surface=selection_stats["skipped_invalid_surface"],
        duplicate_surface_tokens=len(entries) - len(surfaces),
        top_k=cfg.top_k,
        min_similarity=cfg.min_similarity,
    )
    return doc, report, surface_meta
