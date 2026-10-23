from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json
import tempfile
from equinox_forge.hf_adapter import (
    load_adapter, validate_adapter, inspect_hf_model, build_extraction_plan,
    auto_adapter_template, save_adapter, adapter_hash,
)
from equinox_forge.adapter_registry import choose_adapter
from equinox_forge.database import CanonicalDB


def run():
    model = ROOT / "examples" / "hf_synthetic"
    adapter_path = ROOT / "adapters" / "deepseek_v3.json"
    spec = load_adapter(adapter_path)
    assert validate_adapter(spec) == []
    manifest = inspect_hf_model(model)
    assert manifest.model_type == "deepseek_v3"
    assert manifest.vocab_size == 1000
    assert len(manifest.tensor_names) == 4
    plan = build_extraction_plan(manifest, spec)
    assert plan.valid, plan.to_dict()
    roles = {r.role:r for r in plan.resolved_roles}
    assert roles["input_embeddings"].matches == ["model.embed_tokens.weight"]
    assert roles["output_head"].matches == ["lm_head.weight"]
    assert len(roles["transformer_layers"].matches) == 2
    assert "token_embedding_neighbors" in plan.enabled_channels
    assert "output_prediction_priors" in plan.enabled_channels
    assert "runtime_hidden_states" in plan.enabled_channels
    assert plan.adapter_hash == adapter_hash(spec)

    chosen = choose_adapter(model, ROOT / "adapters")
    assert chosen and chosen[1].adapter_id == "hf-deepseek-v3-v1"

    generated = auto_adapter_template(manifest, "synthetic-auto")
    assert not validate_adapter(generated)
    generated_plan = build_extraction_plan(manifest, generated)
    assert generated_plan.valid

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "adapter.json"
        save_adapter(generated, p)
        assert load_adapter(p).adapter_id == "synthetic-auto"
        with CanonicalDB(Path(td)/"forge.sqlite") as db:
            db.register_source_model("synthetic/deepseek", "test", spec.adapter_id, plan.adapter_hash, manifest.model_type, manifest.to_dict())
            db.commit()
            assert db.stats()["registered_models"] == 1
    print(f"FORGE_V02_OK adapter={spec.adapter_id} tensors={len(manifest.tensor_names)} channels={len(plan.enabled_channels)}")

if __name__ == "__main__":
    run()
