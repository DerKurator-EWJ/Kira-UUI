from __future__ import annotations

from dataclasses import dataclass, field, asdict
from fnmatch import fnmatch
import hashlib
import json
from pathlib import Path
from typing import Any

ADAPTER_FORMAT = "equinox-forge-hf-adapter-v1"
PLAN_FORMAT = "equinox-forge-extraction-plan-v1"


@dataclass
class TensorRole:
    role: str
    candidates: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    required: bool = False
    description: str = ""


@dataclass
class ExtractionChannel:
    name: str
    enabled: bool = True
    requires_roles: list[str] = field(default_factory=list)
    runtime_required: bool = False
    description: str = ""


@dataclass
class HFAdapterSpec:
    format: str
    adapter_id: str
    family: str
    model_types: list[str] = field(default_factory=list)
    architectures: list[str] = field(default_factory=list)
    modality: str = "text"
    q_strategy: str = "unicode_character"
    tokenizer: dict[str, Any] = field(default_factory=dict)
    tensor_roles: list[TensorRole] = field(default_factory=list)
    extraction_channels: list[ExtractionChannel] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "adapter_id": self.adapter_id,
            "family": self.family,
            "model_types": self.model_types,
            "architectures": self.architectures,
            "modality": self.modality,
            "q_strategy": self.q_strategy,
            "tokenizer": self.tokenizer,
            "tensor_roles": [asdict(x) for x in self.tensor_roles],
            "extraction_channels": [asdict(x) for x in self.extraction_channels],
            "metadata": self.metadata,
        }


@dataclass
class HFModelManifest:
    model_dir: str
    model_type: str = ""
    architectures: list[str] = field(default_factory=list)
    vocab_size: int | None = None
    hidden_size: int | None = None
    num_hidden_layers: int | None = None
    tokenizer_files: list[str] = field(default_factory=list)
    tensor_names: list[str] = field(default_factory=list)
    tensor_sources: dict[str, str] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tensor_count"] = len(self.tensor_names)
        return d


@dataclass
class ResolvedRole:
    role: str
    matches: list[str]
    required: bool
    ok: bool


@dataclass
class ExtractionPlan:
    format: str
    adapter_id: str
    adapter_hash: str
    model_dir: str
    model_type: str
    architectures: list[str]
    modality: str
    q_strategy: str
    resolved_roles: list[ResolvedRole]
    enabled_channels: list[str]
    unavailable_channels: list[str]
    warnings: list[str]

    @property
    def valid(self) -> bool:
        return not any(r.required and not r.ok for r in self.resolved_roles)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "adapter_id": self.adapter_id,
            "adapter_hash": self.adapter_hash,
            "model_dir": self.model_dir,
            "model_type": self.model_type,
            "architectures": self.architectures,
            "modality": self.modality,
            "q_strategy": self.q_strategy,
            "valid": self.valid,
            "resolved_roles": [asdict(x) for x in self.resolved_roles],
            "enabled_channels": self.enabled_channels,
            "unavailable_channels": self.unavailable_channels,
            "warnings": self.warnings,
        }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_adapter(path: str | Path) -> HFAdapterSpec:
    raw = _read_json(Path(path))
    roles = [TensorRole(**x) for x in raw.get("tensor_roles", [])]
    channels = [ExtractionChannel(**x) for x in raw.get("extraction_channels", [])]
    return HFAdapterSpec(
        format=raw.get("format", ""),
        adapter_id=raw.get("adapter_id", ""),
        family=raw.get("family", ""),
        model_types=list(raw.get("model_types", [])),
        architectures=list(raw.get("architectures", [])),
        modality=raw.get("modality", "text"),
        q_strategy=raw.get("q_strategy", "unicode_character"),
        tokenizer=dict(raw.get("tokenizer", {})),
        tensor_roles=roles,
        extraction_channels=channels,
        metadata=dict(raw.get("metadata", {})),
    )


def save_adapter(spec: HFAdapterSpec, path: str | Path) -> None:
    Path(path).write_text(json.dumps(spec.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_adapter(spec: HFAdapterSpec) -> list[str]:
    errors: list[str] = []
    if spec.format != ADAPTER_FORMAT:
        errors.append(f"unsupported adapter format: {spec.format!r}")
    if not spec.adapter_id.strip():
        errors.append("adapter_id is required")
    if not spec.family.strip():
        errors.append("family is required")
    role_names = [r.role for r in spec.tensor_roles]
    if len(role_names) != len(set(role_names)):
        errors.append("tensor role names must be unique")
    for r in spec.tensor_roles:
        if not r.role.strip():
            errors.append("tensor role without name")
        if not r.candidates and not r.patterns:
            errors.append(f"tensor role {r.role!r} has no candidates/patterns")
    role_set = set(role_names)
    channel_names: set[str] = set()
    for c in spec.extraction_channels:
        if c.name in channel_names:
            errors.append(f"duplicate extraction channel {c.name!r}")
        channel_names.add(c.name)
        missing = [r for r in c.requires_roles if r not in role_set]
        if missing:
            errors.append(f"channel {c.name!r} references unknown roles: {missing}")
    if spec.modality == "text" and spec.q_strategy not in {"unicode_character", "source_token", "byte"}:
        errors.append(f"unsupported text q_strategy: {spec.q_strategy}")
    return errors


def adapter_hash(spec: HFAdapterSpec) -> str:
    raw = json.dumps(spec.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _safetensors_header(path: Path) -> dict[str, Any]:
    """Read only the JSON header; never loads tensor payloads."""
    with path.open("rb") as f:
        first = f.read(8)
        if len(first) != 8:
            raise ValueError(f"invalid safetensors header: {path}")
        n = int.from_bytes(first, "little", signed=False)
        if n <= 0 or n > 128 * 1024 * 1024:
            raise ValueError(f"invalid safetensors header length {n}: {path}")
        raw = f.read(n)
        if len(raw) != n:
            raise ValueError(f"truncated safetensors header: {path}")
    return json.loads(raw.decode("utf-8"))


def inspect_hf_model(model_dir: str | Path) -> HFModelManifest:
    root = Path(model_dir)
    if not root.is_dir():
        raise FileNotFoundError(root)
    cfg = _read_json(root / "config.json")
    tensor_sources: dict[str, str] = {}

    indexes = sorted(root.glob("*.safetensors.index.json"))
    for idx in indexes:
        raw = _read_json(idx)
        for name, shard in raw.get("weight_map", {}).items():
            tensor_sources[name] = shard

    # If no/partial index exists, inspect safetensors headers directly.
    indexed_shards = set(tensor_sources.values())
    for st in sorted(root.glob("*.safetensors")):
        # Indexed models can have hundreds of shards; header reading is still cheap,
        # but skip already-covered shards to keep inspection fast.
        if st.name in indexed_shards and indexes:
            continue
        try:
            header = _safetensors_header(st)
        except Exception:
            continue
        for name in header.keys():
            if name == "__metadata__":
                continue
            tensor_sources.setdefault(name, st.name)

    tokenizer_candidates = [
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "tokenizer.model", "vocab.json", "merges.txt", "spiece.model",
    ]
    tokenizer_files = [name for name in tokenizer_candidates if (root / name).exists()]
    return HFModelManifest(
        model_dir=str(root.resolve()),
        model_type=str(cfg.get("model_type", "")),
        architectures=[str(x) for x in cfg.get("architectures", [])],
        vocab_size=_int_or_none(cfg.get("vocab_size")),
        hidden_size=_int_or_none(cfg.get("hidden_size", cfg.get("d_model"))),
        num_hidden_layers=_int_or_none(cfg.get("num_hidden_layers", cfg.get("n_layer"))),
        tokenizer_files=tokenizer_files,
        tensor_names=sorted(tensor_sources),
        tensor_sources=tensor_sources,
        config=cfg,
    )


def _int_or_none(x: Any) -> int | None:
    try:
        return int(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def _resolve_tensor_role(role: TensorRole, names: list[str]) -> list[str]:
    name_set = set(names)
    out: list[str] = []
    for c in role.candidates:
        if c in name_set and c not in out:
            out.append(c)
    for pat in role.patterns:
        for n in names:
            if fnmatch(n, pat) and n not in out:
                out.append(n)
    return out


def build_extraction_plan(manifest: HFModelManifest, spec: HFAdapterSpec) -> ExtractionPlan:
    errors = validate_adapter(spec)
    if errors:
        raise ValueError("invalid adapter: " + "; ".join(errors))
    warnings: list[str] = []
    if spec.model_types and manifest.model_type not in spec.model_types:
        warnings.append(f"model_type {manifest.model_type!r} not declared by adapter {spec.model_types}")
    if spec.architectures and manifest.architectures and not set(spec.architectures).intersection(manifest.architectures):
        warnings.append(f"architecture mismatch: model={manifest.architectures} adapter={spec.architectures}")
    if not manifest.tokenizer_files:
        warnings.append("no tokenizer files detected in local model directory")
    if not manifest.tensor_names:
        warnings.append("no safetensors manifest detected; weight roles cannot be resolved")

    resolved: list[ResolvedRole] = []
    role_ok: dict[str, bool] = {}
    for role in spec.tensor_roles:
        matches = _resolve_tensor_role(role, manifest.tensor_names)
        ok = bool(matches)
        role_ok[role.role] = ok
        resolved.append(ResolvedRole(role.role, matches, role.required, ok))
        if role.required and not ok:
            warnings.append(f"required tensor role {role.role!r} is unresolved")

    enabled_channels: list[str] = []
    unavailable_channels: list[str] = []
    for ch in spec.extraction_channels:
        if not ch.enabled:
            continue
        if all(role_ok.get(r, False) for r in ch.requires_roles):
            enabled_channels.append(ch.name)
        else:
            unavailable_channels.append(ch.name)

    return ExtractionPlan(
        format=PLAN_FORMAT,
        adapter_id=spec.adapter_id,
        adapter_hash=adapter_hash(spec),
        model_dir=manifest.model_dir,
        model_type=manifest.model_type,
        architectures=manifest.architectures,
        modality=spec.modality,
        q_strategy=spec.q_strategy,
        resolved_roles=resolved,
        enabled_channels=enabled_channels,
        unavailable_channels=unavailable_channels,
        warnings=warnings,
    )


def save_plan(plan: ExtractionPlan, path: str | Path) -> None:
    Path(path).write_text(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def auto_adapter_template(manifest: HFModelManifest, adapter_id: str = "generated-adapter") -> HFAdapterSpec:
    """Generate a conservative starting adapter; user reviews it before extraction."""
    names = manifest.tensor_names
    common_embed = [
        "model.embed_tokens.weight", "transformer.wte.weight", "transformer.word_embeddings.weight",
        "embeddings.word_embeddings.weight",
    ]
    common_head = ["lm_head.weight", "output.weight", "embed_out.weight"]
    embed = [x for x in common_embed if x in names]
    head = [x for x in common_head if x in names]
    return HFAdapterSpec(
        format=ADAPTER_FORMAT,
        adapter_id=adapter_id,
        family=manifest.model_type or "unknown",
        model_types=[manifest.model_type] if manifest.model_type else [],
        architectures=manifest.architectures,
        modality="text",
        q_strategy="unicode_character",
        tokenizer={"required": True, "mode": "huggingface-local"},
        tensor_roles=[
            TensorRole("input_embeddings", candidates=embed or common_embed, required=True,
                       description="Input token embedding matrix; source evidence only, never direct EQX weights."),
            TensorRole("output_head", candidates=head or common_head, required=False,
                       description="Language-model output head for prediction evidence."),
            TensorRole("transformer_layers", patterns=["model.layers.*.*", "transformer.h.*.*"], required=False,
                       description="Optional structural/runtime activation source; not directly copied into Equinox."),
        ],
        extraction_channels=[
            ExtractionChannel("token_embedding_neighbors", True, ["input_embeddings"], False,
                              "Sparse semantic-neighbour evidence from token embeddings."),
            ExtractionChannel("output_prediction_priors", True, ["output_head"], False,
                              "Teacher next-token prior evidence."),
            ExtractionChannel("runtime_hidden_states", True, [], True,
                              "Requires executing the teacher model; no static tensor role alone is sufficient."),
        ],
        metadata={"generated_from": manifest.model_dir, "review_required": True},
    )
