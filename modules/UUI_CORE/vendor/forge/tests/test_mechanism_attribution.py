from pathlib import Path
import sys, tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from equinox_forge.teacher_behavior import TeacherCandidate, BehaviourProbeConfig
from equinox_forge.runtime_cognition import RuntimeCognitionPolicy
from equinox_forge.generalization_ladder import LadderTrainCase, LadderEvalCase
from equinox_forge.mechanism_attribution import compile_and_benchmark_mechanism_attribution


class ScriptedAttributionTeacher:
    backend_id='scripted-attribution-teacher'
    def manifest(self): return {'backend_id':self.backend_id,'fixture':True}
    def next_token(self,prompt,top_k,temperature):
        p=prompt.strip()
        def rows(items):
            return [TeacherCandidate(i+1,s,s,prob,4.0-i) for i,(s,prob) in enumerate(items[:top_k])]
        exact={
            'Hund kann':[(' bellen',.78),(' laufen',.14),(' schlafen',.08)],
            'Mein Hund kann':[(' bellen',.78),(' laufen',.14),(' schlafen',.08)],
            'Hund jagt Maus':[(' heute',.72),(' schnell',.18),(' draußen',.10)],
            'Maus sieht Hund':[(' heute',.72),(' ruhig',.18),(' draußen',.10)],
            'Wolf ist ein':[(' Tier',.82),(' Wesen',.12),(' Haustier',.06)],
            'Hund wohnt in':[(' Haus',.80),(' Stadt',.12),(' Nähe',.08)],
            'Hund ist ein':[(' Tier',.82),(' Haustier',.12),(' Wesen',.06)],
            'Katze ist ein':[(' Tier',.82),(' Haustier',.12),(' Wesen',.06)],
            'Katze kann':[(' miauen',.78),(' schlafen',.14),(' laufen',.08)],
            'Katze jagt Maus':[(' heute',.72),(' schnell',.18),(' draußen',.10)],
            'Hund sieht Katze':[(' heute',.72),(' ruhig',.18),(' draußen',.10)],
            'Haus ist':[(' groß',.76),(' alt',.16),(' nah',.08)],
            'Vogel fliegt':[(' hoch',.76),(' schnell',.16),(' weit',.08)],
            'Katze wohnt':[(' hier',.76),(' ruhig',.16),(' allein',.08)],
            'Vogel fliegt in':[(' Luft',.76),(' Höhe',.16),(' Ferne',.08)],
        }
        if p in exact: return rows(exact[p])
        raise KeyError(p)


def fixture():
    train=[
        LadderTrainCase('Hund ist ein','taxonomy'), LadderTrainCase('Katze ist ein','taxonomy'),
        LadderTrainCase('Hund kann','ability'), LadderTrainCase('Katze kann','ability'),
        LadderTrainCase('Katze jagt Maus','hunting'), LadderTrainCase('Hund sieht Katze','perception'),
        LadderTrainCase('Haus ist','property'), LadderTrainCase('Vogel fliegt','flight'),
        LadderTrainCase('Katze wohnt','residence-bare'), LadderTrainCase('Vogel fliegt in','motion-preposition'),
    ]
    evaluation=[
        LadderEvalCase('Hund kann',0,'ability','exact replay control'),
        LadderEvalCase('Mein Hund kann',1,'ability','new prefix, known predictive suffix'),
        LadderEvalCase('Hund jagt Maus',2,'hunting','known words + local recombination'),
        LadderEvalCase('Maus sieht Hund',3,'perception-reversed','known lexicon, unseen local order'),
        LadderEvalCase('Wolf ist ein',4,'taxonomy','novel entity in known relational frame'),
        LadderEvalCase('Hund wohnt in',5,'residence-preposition','entire relation family held out'),
    ]
    return train,evaluation


with tempfile.TemporaryDirectory() as td:
    cfg=BehaviourProbeConfig(top_k=3,min_probability=0.0,evidence_confidence=0.95)
    policy=RuntimeCognitionPolicy(max_recall_nodes=96,max_recall_depth=3,max_predictions=3,min_prediction_score=0.0,enable_support_gated_recalled_prediction=False)
    train,evaluation=fixture()
    out=Path(td)/'attribution'
    report=compile_and_benchmark_mechanism_attribution(
        ScriptedAttributionTeacher(),train,evaluation,out,'scripted-attribution','1',
        probe_cfg=cfg,cognition_policy=policy,runtime_top_k=3,
    )
    assert abs(report.baseline['top1_retention'] - (5/6)) < 1e-12
    by_name={x.name:x for x in report.ablations}

    # L2-L4 successes require lexical seeding -> parent routing -> recalled prediction.
    for name in ('no_lexical_seeds','no_parent_routing','no_recalled_context_prediction'):
        r=by_name[name]
        assert r.top1_losses == 3
        lost_levels={x.level for x in r.level_deltas if x.top1_losses}
        assert lost_levels == {2,3,4}, (name,lost_levels)

    # Relation recall is not involved in this fixture; teacher behaviour contributes
    # directed prediction edges only. The ablation must therefore be neutral.
    assert by_name['no_relation_recall'].top1_losses == 0
    assert by_name['no_transition_prediction'].top1_losses == 0

    # Suffix routing alone is redundant because lexical->parent can reconstruct L1.
    assert by_name['no_suffix_context'].top1_losses == 0
    # Remove suffix + parent (or lexical) and L1 now collapses too, exposing redundancy.
    combo=by_name['no_suffix_and_parent_routing']
    assert combo.top1_losses == 4
    assert {x.level for x in combo.level_deltas if x.top1_losses} == {1,2,3,4}

    # Direct context prediction is necessary only for L0/L1 in this fixture.
    direct=by_name['no_direct_context_prediction']
    assert direct.top1_losses == 2
    assert {x.level for x in direct.level_deltas if x.top1_losses} == {0,1}

    # v0.21 must not compile evaluation reference evidence into the BASE.
    assert (out/'ladder'/'evaluation_only'/'ladder-teacher-reference.eqir.json').exists()
    assert (out/'mechanism-attribution-report.json').exists()
    assert report.level_mechanism_matrix['L2']['no_parent_routing'] == 1.0
    assert report.level_mechanism_matrix['L5']['no_parent_routing'] == 0.0

print('FORGE_V21_MECHANISM_ATTRIBUTION_OK')
