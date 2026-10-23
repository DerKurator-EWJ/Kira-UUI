from pathlib import Path
import sys, tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from equinox_forge.teacher_behavior import TeacherCandidate, BehaviourProbeConfig
from equinox_forge.runtime_cognition import RuntimeCognitionPolicy
from equinox_forge.behaviour_retention import compile_and_benchmark_teacher_behaviour


class ScriptedRetentionTeacher:
    backend_id='scripted-retention-teacher'
    def manifest(self):
        return {'backend_id':self.backend_id,'fixture':True}
    def next_token(self,prompt,top_k,temperature):
        table={
            'Der Hund ist ein':[
                TeacherCandidate(1,'Tier',' Tier',0.70,4.0),
                TeacherCandidate(2,'Haustier',' Haustier',0.20,2.0),
                TeacherCandidate(3,'Tierchen',' Tierchen',0.10,1.0),
            ],
            'Der Hund kann':[
                TeacherCandidate(4,'bellen',' bellen',0.80,5.0),
                TeacherCandidate(5,'laufen',' laufen',0.15,2.0),
                TeacherCandidate(6,'schlafen',' schlafen',0.05,0.5),
            ],
            'Die Katze kann':[
                TeacherCandidate(7,'miauen',' miauen',0.75,4.5),
                TeacherCandidate(6,'schlafen',' schlafen',0.20,2.0),
                TeacherCandidate(5,'laufen',' laufen',0.05,0.5),
            ],
        }
        return table[prompt][:top_k]


with tempfile.TemporaryDirectory() as td:
    cfg=BehaviourProbeConfig(top_k=3,min_probability=0.0,evidence_confidence=0.95)
    policy=RuntimeCognitionPolicy(max_recall_nodes=48,max_recall_depth=2,max_predictions=3,min_prediction_score=0.00001)
    report=compile_and_benchmark_teacher_behaviour(
        ScriptedRetentionTeacher(),
        ['Der Hund ist ein','Der Hund kann','Die Katze kann'],
        Path(td)/'retention', 'scripted-retention','1',
        probe_cfg=cfg,cognition_policy=policy,runtime_top_k=3,
    )
    assert report.cases == 3
    assert report.evaluation_scope == 'compiled_probe_replay'
    assert report.generalization_claim is False
    assert report.runtime_scores_are_calibrated_probabilities is False
    assert report.top1_retention == 1.0, report.to_dict()
    # Direct replay isolates compiler/pack retention and should be effectively exact.
    assert report.direct_replay['top1_retention'] == 1.0, report.to_dict()
    assert report.direct_replay['mean_teacher_topk_recall'] == 1.0
    assert report.direct_replay['mean_teacher_probability_mass_retained_in_runtime_topk'] == 1.0
    assert report.direct_replay['mean_ndcg_at_k'] > 0.999999
    assert report.direct_replay['mean_conditional_js_divergence'] < 1e-10
    # Full runtime recall may introduce related predictions; top-1 must remain intact.
    assert 0.0 < report.mean_teacher_topk_recall <= 1.0
    assert report.mean_conditional_js_divergence >= 0.0
    assert report.unresolved_contexts == 0
    assert report.no_runtime_predictions == 0
    assert report.pack_bytes > 0 and report.pack_nodes > 0 and report.pack_relations == 9
    assert report.compiled_prediction_evidence == 9
    out=Path(td)/'retention'
    assert (out/'retention.eqbase').exists()
    assert (out/'retention.sqlite').exists()
    assert (out/'teacher-behaviour.eqir.json').exists()
    assert (out/'teacher-probe-report.json').exists()
    assert (out/'retention-report.json').exists()

print('FORGE_V17_BEHAVIOUR_RETENTION_OK')
