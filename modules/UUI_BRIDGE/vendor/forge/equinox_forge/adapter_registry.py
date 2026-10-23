from __future__ import annotations
from pathlib import Path
from .hf_adapter import HFAdapterSpec, load_adapter, inspect_hf_model, build_extraction_plan


def discover_adapters(directory: str | Path) -> list[Path]:
    root = Path(directory)
    return sorted(root.glob("*.json")) if root.exists() else []


def choose_adapter(model_dir: str | Path, adapter_dir: str | Path) -> tuple[Path, HFAdapterSpec] | None:
    manifest = inspect_hf_model(model_dir)
    scored: list[tuple[int, str, Path, HFAdapterSpec]] = []
    for p in discover_adapters(adapter_dir):
        try:
            spec = load_adapter(p)
            plan = build_extraction_plan(manifest, spec)
        except Exception:
            continue
        score = 0
        if manifest.model_type and manifest.model_type in spec.model_types:
            score += 100
        if manifest.architectures and set(manifest.architectures).intersection(spec.architectures):
            score += 50
        score += sum(10 for r in plan.resolved_roles if r.required and r.ok)
        score += sum(1 for r in plan.resolved_roles if r.ok)
        scored.append((score, spec.adapter_id, p, spec))
    if not scored:
        return None
    scored.sort(key=lambda x: (-x[0], x[1]))
    _, _, p, spec = scored[0]
    return p, spec
