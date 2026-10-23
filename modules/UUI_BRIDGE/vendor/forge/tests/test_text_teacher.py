from pathlib import Path
import sys, tempfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from equinox_forge.hf_adapter import load_adapter, inspect_hf_model
from equinox_forge.text_teacher import TextExtractionConfig, extract_text_teacher, load_local_tokenizer_vocab
from equinox_forge.validate import validate_document
from equinox_forge.database import CanonicalDB
from equinox_forge.compiler import aggregate_relations


def run():
    model = ROOT / "examples" / "hf_teacher_synthetic"
    adapter = load_adapter(ROOT / "adapters" / "deepseek_v3.json")
    manifest = inspect_hf_model(model)
    assert manifest.tensor_sources["model.embed_tokens.weight"] == "model.safetensors"
    vocab = load_local_tokenizer_vocab(model)
    assert len(vocab) == 13
    cfg = TextExtractionConfig(max_tokens=32, top_k=2, min_similarity=0.60, batch_size=3)
    doc, report, surface_meta = extract_text_teacher(manifest,adapter,"synthetic/semantic","v1",cfg)
    issues=validate_document(doc)
    assert not [x for x in issues if x.severity=="error"], issues
    assert report.selected_teacher_tokens == 11  # <unk> special and ! punctuation-only skipped
    assert report.surface_nodes == 11
    surfaces={n.surface:n for n in doc.nodes if n.kind=="surface"}
    assert "Hund" in surfaces and "Hunde" in surfaces and "Tier" in surfaces
    assert "!" not in surfaces
    # Synthetic vector fixture should recover these intended close pairs.
    pairs={(e.from_id,e.to_id) for e in doc.evidence}
    assert (surfaces["Hund"].node_id,surfaces["Hunde"].node_id) in pairs
    assert (surfaces["Auto"].node_id,surfaces["Wagen"].node_id) in pairs
    assert (surfaces["laufen"].node_id,surfaces["rennen"].node_id) in pairs
    with tempfile.TemporaryDirectory() as td:
        with CanonicalDB(Path(td)/"forge.sqlite") as db:
            for n in sorted(doc.nodes,key=lambda n:(n.level,n.node_id)): db.upsert_node(n)
            for e in doc.evidence: db.add_evidence(e)
            # Repeat exact same extraction: evidence fingerprint must make ingest idempotent.
            for e in doc.evidence: db.add_evidence(e)
            for surface, entries in surface_meta.items():
                node=surfaces[surface]
                for entry in entries: db.alias("synthetic/semantic","v1",f"token:{entry.token_id}",node.node_id,{"raw_token":entry.raw_token})
            db.commit()
            stats=db.stats()
            assert stats["evidence"] == len(doc.evidence), (stats,len(doc.evidence))
            compiled=aggregate_relations(db.db)
            assert compiled
    print(f"FORGE_V03_OK tokens={report.selected_teacher_tokens} surfaces={report.surface_nodes} q={report.q_nodes} evidence={report.evidence_records}")

if __name__ == "__main__": run()
