from pathlib import Path
import tempfile, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from equinox_forge.teacher_behavior import TeacherCandidate, BehaviourProbeConfig
from equinox_forge.runtime_cognition import RuntimeCognitionPolicy, RuntimeCognition
from equinox_forge.runtime_pack import load_base_pack
from equinox_forge.frame_induction import FrameInductionPolicy
from equinox_forge.frame_schema_benchmark import FrameEvalCase, compile_and_benchmark_frame_schemas


class ScriptedFrameTeacher:
    backend_id='scripted-frame-teacher'
    def manifest(self): return {'backend_id':self.backend_id,'fixture':True}
    def next_token(self,prompt,top_k,temperature):
        p=prompt.strip()
        table={
            'Hund ist ein':[(' Tier',.82),(' Haustier',.12),(' Wesen',.06)],
            'Katze ist ein':[(' Tier',.82),(' Haustier',.12),(' Wesen',.06)],
            # Deliberately non-overlapping outputs: [SLOT] kann must not be induced.
            'Hund kann':[(' bellen',.80),(' rennen',.12),(' springen',.08)],
            'Katze kann':[(' miauen',.80),(' schnurren',.12),(' klettern',.08)],
            # Held out: two intended positive transfers and one counterexample.
            'Wolf ist ein':[(' Tier',.84),(' Wesen',.10),(' Raubtier',.06)],
            'Fuchs ist ein':[(' Tier',.84),(' Wesen',.10),(' Raubtier',.06)],
            'Mein Wolf ist ein':[(' Tier',.84),(' Wesen',.10),(' Raubtier',.06)],
            'Stein ist ein':[(' Objekt',.80),(' Material',.12),(' Ding',.08)],
        }
        vals=table[p][:top_k]
        return [TeacherCandidate(i+1,s,s,prob,4.0-i) for i,(s,prob) in enumerate(vals)]


with tempfile.TemporaryDirectory() as td:
    out=Path(td)/'frame'
    cfg=BehaviourProbeConfig(top_k=3,min_probability=0.0,evidence_confidence=0.95)
    runtime=RuntimeCognitionPolicy(
        max_recall_nodes=96,max_recall_depth=3,max_predictions=3,min_prediction_score=0.0,
    )
    report=compile_and_benchmark_frame_schemas(
        ScriptedFrameTeacher(),
        ['Hund ist ein','Katze ist ein','Hund kann','Katze kann'],
        [
            FrameEvalCase('Wolf ist ein',True,'novel animal filler'),
            FrameEvalCase('Fuchs ist ein',True,'second novel animal filler'),
            FrameEvalCase('Mein Wolf ist ein',True,'new outer prefix plus frame suffix'),
            FrameEvalCase('Stein ist ein',False,'counterexample: open slot must expose overgeneralization'),
        ],
        out,'scripted-frame','1',probe_cfg=cfg,
        frame_policy=FrameInductionPolicy(min_target_context_fraction=1.0),
        cognition_policy=runtime,runtime_top_k=3,
    )
    patterns={x['pattern'] for x in report.frame_induction['frames']}
    assert '[SLOT_0] ist ein' in patterns, patterns
    assert '[SLOT_0] kann' not in patterns, patterns
    assert report.induced_frames == 1

    # With every other discovery/routing mechanism disabled, explicit frame matching
    # transfers the two animal examples to the three novel positive contexts.
    assert abs(report.frame_only['top1_retention'] - 0.75) < 1e-12
    assert report.no_frame['top1_retention'] == 0.0
    assert report.frame_applicability_recall == 1.0
    assert abs(report.frame_applicability_precision - 0.75) < 1e-12

    detail={x.prompt:x for x in report.details}
    assert detail['Wolf ist ein'].frame_only_top1 == 'Tier'
    assert detail['Fuchs ist ein'].frame_only_top1 == 'Tier'
    assert detail['Mein Wolf ist ein'].frame_only_top1 == 'Tier'
    # Critical negative result: an untyped open slot also fires for Stein.
    assert detail['Stein ist ein'].frame_matched is True
    assert detail['Stein ist ein'].frame_only_top1 == 'Tier'
    assert detail['Stein ist ein'].teacher_top1 == 'Objekt'

    # Evaluation-only fillers were never inserted into the compiled BASE.
    pack=load_base_pack(out/'train-with-frames.eqbase')
    cog=RuntimeCognition(pack,runtime)
    assert 'Wolf' not in cog.index.surface_to_nodes
    assert 'Fuchs' not in cog.index.surface_to_nodes
    assert 'Stein' not in cog.index.surface_to_nodes

    assert (out/'evaluation_only'/'frame-eval-teacher-reference.eqir.json').exists()
    assert (out/'induced-frames.eqir.json').exists()
    assert (out/'frame-schema-benchmark-report.json').exists()

print('FORGE_V22_FRAME_SCHEMA_INDUCTION_OK')
