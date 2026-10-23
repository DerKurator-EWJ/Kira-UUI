from pathlib import Path
import sys, tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from equinox_forge.teacher_behavior import TeacherCandidate, BehaviourProbeConfig
from equinox_forge.runtime_cognition import RuntimeCognitionPolicy
from equinox_forge.generalization import compile_and_benchmark_heldout_generalization
from equinox_forge.runtime_pack import load_base_pack


class ScriptedGeneralizationTeacher:
    backend_id='scripted-generalization-teacher'
    def manifest(self):
        return {'backend_id':self.backend_id,'fixture':True}
    def next_token(self,prompt,top_k,temperature):
        def rows(items):
            return [TeacherCandidate(i+1,s,s,prob,4.0-i) for i,(s,prob) in enumerate(items[:top_k])]
        if prompt.endswith('Hund ist ein'):
            return rows([(' Tier',0.70),(' Haustier',0.20),(' Wesen',0.10)])
        if prompt.endswith('Hund kann'):
            return rows([(' bellen',0.70),(' laufen',0.20),(' schlafen',0.10)])
        if prompt.endswith('Katze kann'):
            return rows([(' miauen',0.70),(' schlafen',0.20),(' laufen',0.10)])
        if prompt.endswith('Vogel kann'):
            return rows([(' fliegen',0.70),(' laufen',0.20),(' schlafen',0.10)])
        raise KeyError(prompt)


with tempfile.TemporaryDirectory() as td:
    cfg=BehaviourProbeConfig(top_k=3,min_probability=0.0,evidence_confidence=0.95)
    policy=RuntimeCognitionPolicy(max_recall_nodes=64,max_recall_depth=3,max_predictions=3,min_prediction_score=0.0)
    train=['Hund ist ein','Hund kann','Katze kann']
    heldout=['Der Hund ist ein','Mein Hund kann','Die Katze kann','Vogel kann']
    out=Path(td)/'gen'
    report=compile_and_benchmark_heldout_generalization(
        ScriptedGeneralizationTeacher(), train, heldout, out, 'scripted-generalization','1',
        probe_cfg=cfg,cognition_policy=policy,runtime_top_k=3,
    )
    assert report.exact_split_overlap == 0
    assert report.heldout_exact_context_nodes_in_pack == 0
    assert report.train_replay['top1_retention'] == 1.0
    assert report.heldout_proper_suffix_transfer_rate == 0.75, report.heldout_proper_suffix_transfer_rate
    assert report.heldout_lexical_only_rate == 0.25, report.heldout_lexical_only_rate
    assert report.heldout_full_runtime['top1_retention'] == 0.75, report.heldout_full_runtime
    assert report.heldout_teacher_top1_known_in_pack_rate == 0.75
    assert report.generalization_claim is False
    assert report.heldout_transfer_claim is True
    assert (out/'generalization-report.json').exists()
    assert (out/'split-manifest.json').exists()
    assert (out/'evaluation_only'/'heldout-teacher-reference.eqir.json').exists()

    pack=load_base_pack(out/'train.eqbase')
    surfaces={(n.surface,n.kind) for n in pack.nodes if n.surface}
    assert ('Der Hund ist ein','template') not in surfaces
    assert ('Mein Hund kann','template') not in surfaces
    assert ('Die Katze kann','template') not in surfaces
    assert ('Vogel kann','template') not in surfaces

    # Exact prompt leakage is a hard error by default.
    try:
        compile_and_benchmark_heldout_generalization(
            ScriptedGeneralizationTeacher(), train, ['Hund kann'], Path(td)/'leak',
            'scripted-generalization','1',probe_cfg=cfg,cognition_policy=policy,
        )
        raise AssertionError('expected leakage error')
    except ValueError as exc:
        assert 'leakage' in str(exc)

print('FORGE_V19_HELDOUT_GENERALIZATION_OK')
