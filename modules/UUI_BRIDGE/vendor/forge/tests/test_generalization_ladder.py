from pathlib import Path
import sys, tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from equinox_forge.teacher_behavior import TeacherCandidate, BehaviourProbeConfig
from equinox_forge.runtime_cognition import RuntimeCognitionPolicy
from equinox_forge.generalization_ladder import (
    LadderTrainCase, LadderEvalCase, compile_and_benchmark_generalization_ladder
)


class ScriptedLadderTeacher:
    backend_id='scripted-ladder-teacher'
    def manifest(self):
        return {'backend_id':self.backend_id,'fixture':True}
    def next_token(self,prompt,top_k,temperature):
        p=prompt.strip()
        def rows(items):
            return [TeacherCandidate(i+1,s,s,prob,4.0-i) for i,(s,prob) in enumerate(items[:top_k])]
        # Evaluation overrides first so held-out semantics can differ from nearby train contexts.
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


with tempfile.TemporaryDirectory() as td:
    cfg=BehaviourProbeConfig(top_k=3,min_probability=0.0,evidence_confidence=0.95)
    policy=RuntimeCognitionPolicy(max_recall_nodes=96,max_recall_depth=3,max_predictions=3,min_prediction_score=0.0,enable_support_gated_recalled_prediction=False)
    train=[
        LadderTrainCase('Hund ist ein','taxonomy'),
        LadderTrainCase('Katze ist ein','taxonomy'),
        LadderTrainCase('Hund kann','ability'),
        LadderTrainCase('Katze kann','ability'),
        LadderTrainCase('Katze jagt Maus','hunting'),
        LadderTrainCase('Hund sieht Katze','perception'),
        LadderTrainCase('Haus ist','property'),
        LadderTrainCase('Vogel fliegt','flight'),
        LadderTrainCase('Katze wohnt','residence-bare'),
        LadderTrainCase('Vogel fliegt in','motion-preposition'),
    ]
    evaluation=[
        LadderEvalCase('Hund kann',0,'ability','exact replay control'),
        LadderEvalCase('Mein Hund kann',1,'ability','new prefix, known predictive suffix'),
        LadderEvalCase('Hund jagt Maus',2,'hunting','known words + known jagt Maus bigram in new composition'),
        LadderEvalCase('Maus sieht Hund',3,'perception-reversed','known lexicon, unseen local order'),
        LadderEvalCase('Wolf ist ein',4,'taxonomy','novel entity in known relational frame'),
        LadderEvalCase('Hund wohnt in',5,'residence-preposition','entire relation family held out'),
    ]
    out=Path(td)/'ladder'
    report=compile_and_benchmark_generalization_ladder(
        ScriptedLadderTeacher(),train,evaluation,out,'scripted-ladder','1',
        probe_cfg=cfg,cognition_policy=policy,runtime_top_k=3,
    )
    assert report.train_cases == 10
    assert report.evaluation_cases == 6
    assert report.exact_replay_controls == 1
    assert report.heldout_cases == 5
    assert all(x.structural_contract_ok for x in report.structural_diagnostics), [x.to_dict() for x in report.structural_diagnostics]
    levels={x.level:x for x in report.level_reports}
    assert levels[0].top1_retention == 1.0
    assert levels[1].top1_retention == 1.0
    assert levels[4].top1_retention == 1.0
    # L5 is intentionally beyond current structural support in this fixture.
    assert levels[5].top1_retention == 0.0
    assert report.highest_level_with_any_top1_success is not None
    assert (out/'generalization-ladder-report.json').exists()
    assert (out/'ladder-manifest.json').exists()
    assert (out/'evaluation_only'/'ladder-teacher-reference.eqir.json').exists()

    # Mislabeling L1 as L3 must be rejected because it has a known whole-prompt suffix.
    try:
        compile_and_benchmark_generalization_ladder(
            ScriptedLadderTeacher(),train,[LadderEvalCase('Mein Hund kann',3,'x')],Path(td)/'bad',
            'scripted-ladder','1',probe_cfg=cfg,cognition_policy=policy,
        )
        raise AssertionError('expected structural contract failure')
    except ValueError as exc:
        assert 'contract' in str(exc)

print('FORGE_V20_GENERALIZATION_LADDER_OK')
