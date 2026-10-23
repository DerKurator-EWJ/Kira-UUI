from pathlib import Path
import sys, tempfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from equinox_forge.language_evidence import LanguageEvidenceConfig, build_controlled_language_evidence
from equinox_forge.validate import validate_document
from equinox_forge.database import CanonicalDB
from equinox_forge.compiler import aggregate_relations


def run():
    corpus = [
        "Der Hund rennt. Der Hund läuft. Der Hund rennt. Die Katze rennt. Die Katze läuft.",
        "Ein Hund ist ein Tier. Eine Katze ist ein Tier. Der Hund ist ein Tier.",
    ]
    cfg = LanguageEvidenceConfig(max_context=2, min_phrase_count=2, min_substitution_score=0.10)
    doc, report = build_controlled_language_evidence(corpus, "corpus/german-mini", "v1", cfg)
    issues = validate_document(doc)
    assert not [x for x in issues if x.severity == "error"], issues
    surfaces = {n.surface:n for n in doc.nodes if n.kind == "surface"}
    assert {"Der","Hund","rennt","läuft","Katze","Tier"}.issubset(surfaces)

    # Directed transition should encode empirical conditional probability.
    hr = [e for e in doc.evidence if e.relation_type == "transition" and e.from_id == surfaces["Hund"].node_id and e.to_id == surfaces["rennt"].node_id]
    assert hr and 0.0 < hr[0].score <= 1.0

    # Context template "Der Hund" must predict continuations observed after it.
    ctx = [n for n in doc.nodes if n.kind == "template" and n.surface == "Der Hund"]
    assert len(ctx) == 1
    preds = [e for e in doc.evidence if e.relation_type == "prediction" and e.from_id == ctx[0].node_id]
    assert {e.to_id for e in preds} >= {surfaces["rennt"].node_id, surfaces["läuft"].node_id, surfaces["ist"].node_id}

    # rennt/läuft share the same left/right slots and should become substitution evidence.
    subs = [e for e in doc.evidence if e.relation_type == "substitution" and e.from_id == surfaces["rennt"].node_id and e.to_id == surfaces["läuft"].node_id]
    assert subs and subs[0].score > 0

    # Repeated phrases should be materialized as observations, but not every one-off phrase.
    phrases = {n.surface:n for n in doc.nodes if n.kind == "composite"}
    assert "Der Hund" in phrases

    with tempfile.TemporaryDirectory() as td:
        with CanonicalDB(Path(td)/"forge.sqlite") as db:
            for n in doc.nodes: db.upsert_node(n)
            for e in doc.evidence: db.add_evidence(e)
            for e in doc.evidence: db.add_evidence(e)
            db.commit()
            assert db.stats()["evidence"] == len(doc.evidence)
            rels = aggregate_relations(db.db)
            assert any(r.relation_type == "prediction" for r in rels)
            assert any(r.relation_type == "substitution" for r in rels)
    print(f"FORGE_V04_OK sentences={report.sentences} tokens={report.lexical_tokens} surfaces={report.unique_surfaces} phrases={report.phrase_nodes} transition={report.transition_evidence} prediction={report.prediction_evidence} substitution={report.substitution_evidence}")

if __name__ == "__main__":
    run()
