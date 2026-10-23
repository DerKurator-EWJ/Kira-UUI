from pathlib import Path
import sys, tempfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from equinox_forge.teacher_behavior import (
    TeacherCandidate, BehaviourProbeConfig, extract_teacher_behaviour
)
from equinox_forge.validate import validate_document
from equinox_forge.database import CanonicalDB
from equinox_forge.language_evidence import build_controlled_language_evidence, LanguageEvidenceConfig


class ScriptedBackend:
    backend_id = "scripted-test-teacher"
    def manifest(self): return {"backend_id": self.backend_id}
    def next_token(self, prompt, top_k, temperature):
        table = {
            "Der Hund ist ein": [
                TeacherCandidate(7,"ĠTier"," Tier",0.72,4.2),
                TeacherCandidate(8,"ĠHund"," Hund",0.18,2.1),
                TeacherCandidate(9,"!","!",0.05,0.1),
            ],
            "Die Katze ist ein": [
                TeacherCandidate(7,"ĠTier"," Tier",0.81,4.8),
                TeacherCandidate(10,"ĠHaustier"," Haustier",0.10,1.8),
            ],
        }
        return table[prompt][:top_k]


def main():
    cfg = BehaviourProbeConfig(top_k=3,min_probability=0.06,evidence_confidence=0.93)
    doc, report, probes = extract_teacher_behaviour(
        ScriptedBackend(), ["Der Hund ist ein","Die Katze ist ein"], "teacher-test", "1", cfg
    )
    errors = [i for i in validate_document(doc) if i.severity == "error"]
    assert not errors, errors
    assert report.prompts == 2
    assert report.unique_contexts == 2
    assert report.prediction_evidence == 4
    assert report.skipped_low_probability == 1
    assert any(n.surface == "Tier" for n in doc.nodes)
    assert sum(1 for n in doc.nodes if n.kind == "template") == 2
    tier_e = [e for e in doc.evidence if next(n for n in doc.nodes if n.node_id==e.to_id).surface == "Tier"]
    assert len(tier_e) == 2
    assert all(e.relation_type == "prediction" for e in tier_e)
    assert tier_e[0].metadata["evidence_kind"] == "teacher_behaviour"
    assert abs(tier_e[0].confidence - 0.93) < 1e-9

    # Corpus prediction and teacher behaviour must share the same canonical
    # context/target pair while retaining separate evidence provenance.
    corpus_doc, _ = build_controlled_language_evidence(
        ["Der Hund ist ein Tier. Der Hund ist ein Tier."],
        "corpus-test", "1", LanguageEvidenceConfig(max_context=4,min_phrase_count=99)
    )
    teacher_tier = next(e for e in doc.evidence if e.metadata["prompt"] == "Der Hund ist ein" and next(n for n in doc.nodes if n.node_id==e.to_id).surface == "Tier")
    corpus_tier = next(e for e in corpus_doc.evidence if e.relation_type == "prediction" and e.from_id == teacher_tier.from_id and e.to_id == teacher_tier.to_id)
    assert corpus_tier.source.source_id == "corpus-test"

    with tempfile.TemporaryDirectory() as td:
        with CanonicalDB(Path(td)/"behaviour.sqlite") as db:
            for n in doc.nodes: db.upsert_node(n)
            for e in doc.evidence: db.add_evidence(e)
            db.commit()
            stats1 = db.stats()
            # repeated probe ingestion must be idempotent
            for e in doc.evidence: db.add_evidence(e)
            db.commit()
            stats2 = db.stats()
            assert stats1["evidence"] == stats2["evidence"] == 4
    print("FORGE_V05_BEHAVIOUR_OK")

if __name__ == "__main__": main()
