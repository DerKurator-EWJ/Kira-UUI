from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json
import tempfile

from equinox_forge.database import CanonicalDB
from equinox_forge.language_evidence import LanguageEvidenceConfig, build_controlled_language_evidence
from equinox_forge.macro_discovery import MacroDiscoveryPolicy
from equinox_forge.runtime_compression import MacroPromotionPolicy, compile_runtime_compression
from equinox_forge.runtime_pack import compile_base_pack, load_base_pack
from equinox_forge.runtime_cognition import (
    RuntimeCognition, RuntimeCognitionPolicy, PredictionBenchmarkCase,
    benchmark_predictions, load_prediction_cases,
)
from equinox_forge.ir import RelationType

TEXTS = [
    "Der Hund ist ein Tier.",
    "Der Hund ist ein Tier.",
    "Die Katze ist ein Tier.",
    "Der Hund kann bellen.",
    "Der Hund kann bellen.",
    "Die Katze kann miauen.",
    "Die Katze kann miauen.",
]

with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    db_path = root / "cognition.sqlite"
    pack_path = root / "cognition.eqbase"
    cases_path = root / "cases.jsonl"

    doc, report = build_controlled_language_evidence(
        TEXTS, "corpus-v016", "1",
        LanguageEvidenceConfig(max_context=3, min_phrase_count=2, evidence_confidence=0.9),
    )
    assert report.prediction_evidence > 0
    assert report.transition_evidence > 0

    with CanonicalDB(db_path) as c:
        for n in doc.nodes: c.upsert_node(n)
        for e in doc.evidence: c.add_evidence(e)
        c.commit()
        compression = compile_runtime_compression(
            c.db,
            MacroDiscoveryPolicy(
                min_pattern_size=2, max_pattern_size=4,
                min_occurrences=999, min_parent_structures=999,
                definition_overhead=1.0, min_net_savings=1.0,
                min_compression_ratio=0.0, prune_dominated=True,
            ),
            MacroPromotionPolicy(
                min_realized_occurrences=999, min_realized_parent_structures=999,
                definition_overhead=1.0, min_realized_net_savings=1.0,
                min_realized_compression_ratio=0.0, max_promotions=1,
            ),
            persist=True, materialize=True,
        )
        manifest = compile_base_pack(
            c.db, pack_path, pack_id="v016-language", pack_version=16,
            compression_run_id=compression.compression_run_id,
            include_metadata=False, persist=False, verify_roundtrip=True,
        )
        assert manifest.roundtrip_valid
        assert manifest.relation_count > 0

    pack = load_base_pack(pack_path)
    policy = RuntimeCognitionPolicy(max_recall_nodes=64, max_recall_depth=3, max_predictions=6)
    cog = RuntimeCognition(pack, policy)

    r1 = cog.predict("Der Hund ist ein")
    assert r1.predictions, r1.to_dict()
    assert r1.predictions[0].surface == "Tier", r1.to_dict()
    assert r1.predictions[0].direct_score > 0
    assert any(c.relation_type == RelationType.PREDICTION.value for c in r1.predictions[0].contributions)
    assert r1.resolved_context_ids, "longer prompt should resolve longest known suffix context"
    assert r1.work["recall_nodes"] <= policy.max_recall_nodes

    r2 = cog.predict("Der Hund kann")
    assert r2.predictions[0].surface == "bellen", r2.to_dict()
    r3 = cog.predict("Die Katze kann")
    assert r3.predictions[0].surface == "miauen", r3.to_dict()

    # A single lexical context still produces bounded candidates via transitions and
    # recalled predictive parents, but semantic similarity itself cannot be output.
    r4 = cog.predict("Hund")
    assert len(r4.predictions) <= policy.max_predictions
    for p in r4.predictions:
        assert all(c.relation_type in {RelationType.PREDICTION.value, RelationType.TRANSITION.value}
                   for c in p.contributions)

    cases = [
        PredictionBenchmarkCase("Der Hund ist ein", "Tier"),
        PredictionBenchmarkCase("Der Hund kann", "bellen"),
        PredictionBenchmarkCase("Die Katze kann", "miauen"),
    ]
    bench = benchmark_predictions(pack, cases, policy=policy, top_k=3)
    assert bench.cases == 3
    assert bench.hit_at_1 == 1.0, bench.to_dict()
    assert bench.hit_at_k == 1.0
    assert bench.mean_reciprocal_rank == 1.0
    assert bench.unresolved_contexts == 0
    assert bench.expected_not_in_pack == 0

    cases_path.write_text("\n".join(json.dumps({"context": x.context, "expected": x.expected}, ensure_ascii=False) for x in cases)+"\n", encoding="utf-8")
    loaded_cases = load_prediction_cases([cases_path])
    assert [(x.context, x.expected) for x in loaded_cases] == [(x.context, x.expected) for x in cases]

print("FORGE_V16_RUNTIME_COGNITION_OK")
