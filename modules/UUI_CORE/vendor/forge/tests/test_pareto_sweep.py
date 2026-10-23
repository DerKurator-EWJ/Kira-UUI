from pathlib import Path
import sys, tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from equinox_forge.teacher_behavior import TeacherCandidate, BehaviourProbeConfig
from equinox_forge.runtime_cognition import RuntimeCognitionPolicy
from equinox_forge.pareto_sweep import ParetoSweepConfig, run_behaviour_pareto_sweep


class ScriptedParetoTeacher:
    backend_id='scripted-pareto-teacher'
    def manifest(self):
        return {'backend_id':self.backend_id,'fixture':True}
    def next_token(self,prompt,top_k,temperature):
        table={
            'Der Hund ist ein':[
                TeacherCandidate(1,'Tier',' Tier',0.60,4.0),
                TeacherCandidate(2,'Haustier',' Haustier',0.25,2.0),
                TeacherCandidate(3,'Tierchen',' Tierchen',0.10,1.0),
                TeacherCandidate(4,'Freund',' Freund',0.05,0.5),
            ],
            'Der Hund kann':[
                TeacherCandidate(5,'bellen',' bellen',0.65,5.0),
                TeacherCandidate(6,'laufen',' laufen',0.20,2.0),
                TeacherCandidate(7,'schlafen',' schlafen',0.10,1.0),
                TeacherCandidate(8,'spielen',' spielen',0.05,0.5),
            ],
            'Die Katze kann':[
                TeacherCandidate(9,'miauen',' miauen',0.60,4.5),
                TeacherCandidate(7,'schlafen',' schlafen',0.20,2.0),
                TeacherCandidate(6,'laufen',' laufen',0.15,1.0),
                TeacherCandidate(8,'spielen',' spielen',0.05,0.5),
            ],
        }
        return table[prompt][:top_k]


with tempfile.TemporaryDirectory() as td:
    cfg=BehaviourProbeConfig(top_k=4,min_probability=0.0,evidence_confidence=0.95)
    policy=RuntimeCognitionPolicy(max_recall_nodes=48,max_recall_depth=2,max_predictions=4,min_prediction_score=0.0)
    sweep=ParetoSweepConfig(
        evidence_top_k_values=[1,4],
        min_relation_score_values=[0.0,0.11],
        max_recall_nodes_values=[12,48],
        max_recall_depth_values=[0,2],
        compression_modes=['none','auto'],
        max_points=64,
        runtime_top_k=4,
    )
    report=run_behaviour_pareto_sweep(
        ScriptedParetoTeacher(),
        ['Der Hund ist ein','Der Hund kann','Die Katze kann'],
        Path(td)/'pareto','scripted-pareto','1',
        probe_cfg=cfg,base_cognition_policy=policy,sweep_cfg=sweep,
    )
    assert report.points == 32, report.points
    assert report.generalization_claim is False
    assert report.teacher_top_k == 4 and report.runtime_top_k == 4
    assert report.pareto_storage_ids and report.pareto_runtime_ids and report.pareto_joint_ids
    assert (Path(td)/'pareto'/'pareto-report.json').exists()
    assert (Path(td)/'pareto'/'teacher-behaviour-full.eqir.json').exists()

    rows=report.results
    # Evidence pruning must create a real storage/retention tradeoff.
    k1=[x for x in rows if x.config['evidence_top_k']==1 and x.config['min_relation_score']==0.0 and x.config['compression_mode']=='none']
    k4=[x for x in rows if x.config['evidence_top_k']==4 and x.config['min_relation_score']==0.0 and x.config['compression_mode']=='none']
    assert k1 and k4
    assert min(x.pack_bytes for x in k1) < max(x.pack_bytes for x in k4)
    assert max(x.direct_teacher_mass_retained for x in k1) < min(x.direct_teacher_mass_retained for x in k4)
    assert max(x.direct_top1_retention for x in k1) == 1.0
    assert min(x.direct_top1_retention for x in k4) == 1.0

    # Pruning threshold 0.11 should remove the .10/.05 tails and shrink relation count.
    full=[x for x in rows if x.config['evidence_top_k']==4 and x.config['min_relation_score']==0.0 and x.config['compression_mode']=='none']
    pruned=[x for x in rows if x.config['evidence_top_k']==4 and x.config['min_relation_score']==0.11 and x.config['compression_mode']=='none']
    assert min(x.pack_relations for x in pruned) < max(x.pack_relations for x in full)
    assert max(x.direct_teacher_mass_retained for x in pruned) < min(x.direct_teacher_mass_retained for x in full)

print('FORGE_V18_PARETO_SWEEP_OK')
