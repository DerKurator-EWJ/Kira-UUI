from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Protocol
import math

from .canonical import make_text_surface, make_composite, normalize_surface
from .ir import ForgeDocument, IRNode, RelationEvidence, RelationType, SourceRef, NodeKind, Modality
from .language_evidence import tokenize_text, LanguageEvidenceConfig


BEHAVIOUR_EXTRACTOR_FORMAT = "equinox-forge-teacher-behaviour-v1"


@dataclass(frozen=True)
class TeacherCandidate:
    token_id: int
    raw_token: str
    surface: str
    probability: float
    logit: float | None = None
    special: bool = False


@dataclass(frozen=True)
class TeacherProbe:
    prompt: str
    candidates: list[TeacherCandidate]
    metadata: dict[str, Any] | None = None


class CausalTeacherBackend(Protocol):
    """Minimal runtime boundary for a causal language-model teacher.

    Forge deliberately does not depend on one ML runtime. Any backend that can
    return the teacher's next-token distribution can feed the canonical compiler.
    """

    backend_id: str

    def next_token(self, prompt: str, top_k: int, temperature: float) -> list[TeacherCandidate]: ...
    def manifest(self) -> dict[str, Any]: ...


@dataclass
class BehaviourProbeConfig:
    top_k: int = 8
    temperature: float = 1.0
    min_probability: float = 0.001
    max_surface_chars: int = 64
    evidence_confidence: float = 0.95
    strip_leading_space: bool = True
    include_punctuation_only: bool = False
    include_numeric_only: bool = True
    max_prompt_words: int = 64

    def validate(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be > 0")
        if self.temperature <= 0.0 or not math.isfinite(self.temperature):
            raise ValueError("temperature must be finite and > 0")
        if not 0.0 <= self.min_probability <= 1.0:
            raise ValueError("min_probability must be in [0, 1]")
        if self.max_surface_chars <= 0:
            raise ValueError("max_surface_chars must be > 0")
        if not 0.0 <= self.evidence_confidence <= 1.0:
            raise ValueError("evidence_confidence must be in [0, 1]")
        if self.max_prompt_words <= 0:
            raise ValueError("max_prompt_words must be > 0")


@dataclass
class BehaviourProbeReport:
    format: str
    source_id: str
    source_version: str
    backend_id: str
    prompts: int
    unique_contexts: int
    context_nodes: int
    q_nodes: int
    surface_nodes: int
    candidate_records: int
    prediction_evidence: int
    skipped_special: int
    skipped_low_probability: int
    skipped_invalid_surface: int
    top_k: int
    temperature: float
    min_probability: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _valid_surface(surface: str, cfg: BehaviourProbeConfig) -> bool:
    if not surface or len(surface) > cfg.max_surface_chars:
        return False
    if any(ord(ch) < 32 and ch not in {"\t", "\n"} for ch in surface):
        return False
    stripped = surface.strip()
    if not stripped:
        return False
    if not cfg.include_punctuation_only and not any(ch.isalnum() for ch in stripped):
        return False
    if not cfg.include_numeric_only and stripped.replace(".", "", 1).isdigit():
        return False
    return True


def canonical_candidate_surface(candidate: TeacherCandidate, cfg: BehaviourProbeConfig) -> str:
    surface = normalize_surface(candidate.surface)
    if cfg.strip_leading_space:
        surface = surface.lstrip()
    return surface


def _context_parts(prompt: str, cfg: BehaviourProbeConfig) -> list[str]:
    lang_cfg = LanguageEvidenceConfig(include_punctuation=False)
    sentences = tokenize_text(normalize_surface(prompt), lang_cfg)
    tokens = [tok for sent in sentences for tok in sent]
    if len(tokens) > cfg.max_prompt_words:
        tokens = tokens[-cfg.max_prompt_words:]
    return tokens


def _context_node(prompt: str, cfg: BehaviourProbeConfig) -> tuple[list[IRNode], IRNode]:
    parts = _context_parts(prompt, cfg)
    nodes: dict[str, IRNode] = {}
    surfaces: list[IRNode] = []
    if not parts:
        qs, surface = make_text_surface(normalize_surface(prompt))
        for n in [*qs, surface]: nodes[n.node_id] = n
        return list(nodes.values()), surface
    for part in parts:
        qs, surface = make_text_surface(part)
        for n in [*qs, surface]: nodes[n.node_id] = n
        surfaces.append(surface)
    # Even a one-word prompt receives a context/template node. This prevents
    # behaviour evidence from being confused with corpus transition evidence.
    context = make_composite(
        Modality.TEXT.value,
        surfaces,
        kind=NodeKind.TEMPLATE.value,
        label=" ".join(parts),
    )
    nodes[context.node_id] = context
    return list(nodes.values()), context


def extract_teacher_behaviour(
    backend: CausalTeacherBackend,
    prompts: Iterable[str],
    source_id: str,
    source_version: str,
    cfg: BehaviourProbeConfig | None = None,
) -> tuple[ForgeDocument, BehaviourProbeReport, list[TeacherProbe]]:
    """Run controlled next-token probes and compile them into canonical IR.

    The relation score is the teacher probability, not an Equinox-combined weight.
    Provenance keeps rank/token/logit/backend metadata so later aggregation can be
    changed without rerunning the teacher.
    """
    cfg = cfg or BehaviourProbeConfig()
    cfg.validate()
    src = SourceRef(source_id, source_version, "teacher_next_token_probe")
    node_map: dict[str, IRNode] = {}
    evidence: list[RelationEvidence] = []
    probe_records: list[TeacherProbe] = []
    context_ids: set[str] = set()
    skipped_special = skipped_low = skipped_invalid = 0
    candidate_records = 0
    seen_prompts = 0

    for raw_prompt in prompts:
        prompt = normalize_surface(str(raw_prompt)).strip("\r\n")
        if not prompt.strip():
            continue
        seen_prompts += 1
        context_nodes, context = _context_node(prompt, cfg)
        for n in context_nodes: node_map[n.node_id] = n
        context_ids.add(context.node_id)

        candidates = backend.next_token(prompt, cfg.top_k, cfg.temperature)
        accepted: list[TeacherCandidate] = []
        for rank, cand in enumerate(candidates[:cfg.top_k], start=1):
            candidate_records += 1
            if cand.special:
                skipped_special += 1
                continue
            if cand.probability < cfg.min_probability:
                skipped_low += 1
                continue
            surface = canonical_candidate_surface(cand, cfg)
            if not _valid_surface(surface, cfg):
                skipped_invalid += 1
                continue
            qs, target = make_text_surface(surface)
            for n in [*qs, target]: node_map[n.node_id] = n
            evidence.append(RelationEvidence(
                from_id=context.node_id,
                to_id=target.node_id,
                relation_type=RelationType.PREDICTION.value,
                score=float(cand.probability),
                confidence=float(cfg.evidence_confidence),
                source=src,
                metadata={
                    "evidence_kind": "teacher_behaviour",
                    "backend_id": backend.backend_id,
                    "prompt": prompt,
                    "rank": rank,
                    "teacher_token_id": int(cand.token_id),
                    "raw_token": cand.raw_token,
                    "decoded_surface": cand.surface,
                    "canonical_surface": surface,
                    "teacher_probability": float(cand.probability),
                    "teacher_logit": None if cand.logit is None else float(cand.logit),
                    "temperature": float(cfg.temperature),
                    "top_k": int(cfg.top_k),
                },
            ))
            accepted.append(cand)
        probe_records.append(TeacherProbe(prompt, accepted, {"backend_id": backend.backend_id}))

    nodes = sorted(node_map.values(), key=lambda n: (n.level, n.node_id))
    doc = ForgeDocument(nodes=nodes, evidence=evidence)
    report = BehaviourProbeReport(
        format=BEHAVIOUR_EXTRACTOR_FORMAT,
        source_id=source_id,
        source_version=source_version,
        backend_id=backend.backend_id,
        prompts=seen_prompts,
        unique_contexts=len(context_ids),
        context_nodes=sum(1 for n in nodes if n.kind == NodeKind.TEMPLATE.value),
        q_nodes=sum(1 for n in nodes if n.kind == NodeKind.QUANT.value),
        surface_nodes=sum(1 for n in nodes if n.kind == NodeKind.SURFACE.value),
        candidate_records=candidate_records,
        prediction_evidence=len(evidence),
        skipped_special=skipped_special,
        skipped_low_probability=skipped_low,
        skipped_invalid_surface=skipped_invalid,
        top_k=cfg.top_k,
        temperature=cfg.temperature,
        min_probability=cfg.min_probability,
    )
    return doc, report, probe_records


def load_prompts(paths: Iterable[str | Path]) -> list[str]:
    prompts: list[str] = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            text = line.strip("\ufeff\r\n")
            if not text.strip() or text.lstrip().startswith("#"):
                continue
            prompts.append(text)
    return prompts


class TransformersCausalLMBackend:
    """Optional local Hugging Face backend.

    Dependencies are imported lazily so the Forge compiler remains usable without
    PyTorch/Transformers. `local_files_only=True` is intentional: Forge never pulls
    a model from the network behind the user's back.
    """

    backend_id = "huggingface-transformers-causal-lm"

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str = "cpu",
        trust_remote_code: bool = False,
        dtype: str = "auto",
    ):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except Exception as exc:  # pragma: no cover - depends on optional runtime
            raise RuntimeError(
                "Teacher behaviour probing requires the optional HF runtime. "
                "Install with: pip install -e '.[hf-runtime]'"
            ) from exc
        self._torch = torch
        self.model_dir = str(Path(model_dir).resolve())
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
        kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": trust_remote_code,
        }
        if dtype != "auto":
            dtype_obj = getattr(torch, dtype, None)
            if dtype_obj is None:
                raise ValueError(f"unknown torch dtype: {dtype}")
            kwargs["torch_dtype"] = dtype_obj
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, **kwargs)
        self.model.to(device)
        self.model.eval()

    def manifest(self) -> dict[str, Any]:
        cfg = getattr(self.model, "config", None)
        return {
            "backend_id": self.backend_id,
            "model_dir": self.model_dir,
            "device": self.device,
            "model_type": getattr(cfg, "model_type", "") if cfg is not None else "",
            "vocab_size": getattr(cfg, "vocab_size", None) if cfg is not None else None,
        }

    def next_token(self, prompt: str, top_k: int, temperature: float) -> list[TeacherCandidate]:
        torch = self._torch
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1].float()
            scaled = logits / float(temperature)
            probs = torch.softmax(scaled, dim=-1)
            k = min(int(top_k), int(probs.shape[-1]))
            values, indices = torch.topk(probs, k=k)
        special_ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])
        out: list[TeacherCandidate] = []
        for probability, token_id in zip(values.tolist(), indices.tolist()):
            raw_token = self.tokenizer.convert_ids_to_tokens(int(token_id))
            surface = self.tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            out.append(TeacherCandidate(
                token_id=int(token_id),
                raw_token=str(raw_token),
                surface=str(surface),
                probability=float(probability),
                logit=float(logits[int(token_id)].item()),
                special=int(token_id) in special_ids,
            ))
        return out
