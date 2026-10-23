"""Python implementation of Kotlin DecisionMatrix v0.1 for Numeric Core integration."""
from dataclasses import dataclass
import math

LABELS = {"OK":"OK.","MMH":"Mmh.","GEIL":"Geil!", "HAEH":"Häh?!", "WAS":"Was?!",
          "NOTGEIL":"Notgeil!", "FUCK":"Fuck!", "UUH":"Uuh!", "BORING":"Boring.",
          "SURPRISE":"Surprise, Motherfucker!", "GUILTY":"Guilty.", "EGAL":"Egal.", "CANTOR":"Cantor."}
PRECEDENCE = ('CANTOR','NOTGEIL','FUCK','WAS','GUILTY','SURPRISE','HAEH','UUH','GEIL','MMH','BORING','EGAL','OK')

@dataclass(frozen=True)
class EvaluationInput:
    expectation_confidence:float
    surprise:float
    success:float
    novelty:float
    relevance:float
    information_gain:float
    valence:float
    repetition:float=0.0
    contradiction_scope:float=0.0
    suspicious_recurrence:float=0.0
    uncertainty:float=0.0
    def __post_init__(self):
        for name,value in vars(self).items():
            lower = -1.0 if name == 'valence' else 0.0
            if not math.isfinite(value) or not lower <= value <= 1.0:
                raise ValueError(f'{name} outside [{lower},1]')

@dataclass(frozen=True)
class Evaluation:
    primary:str
    signals:tuple
    investigate_priority:float
    repeat_priority:float
    evidence_review_required:bool
    @property
    def utterance(self):return LABELS[self.primary]

class DecisionMatrix:
    def evaluate(self,x):
        surprise=x.surprise>=0.65
        contradiction=surprise and x.expectation_confidence>=0.75
        good=x.valence>=0.55 and x.success>=0.55
        bad=x.valence<=-0.55 or (x.success<=0.25 and x.relevance>=0.6)
        signals=set()
        if x.relevance<=0.15 and x.information_gain<=0.15:signals.add('EGAL')
        if x.repetition>=0.8 and x.information_gain<=0.15:signals.add('BORING')
        if x.information_gain>=0.7 and x.uncertainty<=0.35:signals.add('UUH')
        if x.novelty>=0.75 and x.information_gain>=0.3 and surprise:signals.add('SURPRISE')
        if x.suspicious_recurrence>=0.8 and contradiction:signals.add('GUILTY')
        if surprise and x.contradiction_scope>=0.9 and x.expectation_confidence>=0.9:signals.add('CANTOR')
        if surprise and x.contradiction_scope>=0.7:signals.add('WAS')
        if surprise:signals.add('HAEH')
        if good:signals.add('GEIL')
        if bad:signals.add('FUCK')
        if good and surprise:signals.add('NOTGEIL')
        if x.uncertainty>=0.55 or (x.novelty>=0.6 and not good and not bad):signals.add('MMH')
        if not signals:signals.add('OK')
        investigate=min(1.,max(0.,0.4*x.surprise*x.expectation_confidence+0.3*x.information_gain+0.2*x.novelty+0.1*x.relevance))
        repeat=min(1.,max(0.,0.55*x.success+0.25*max(0.,x.valence)+0.2*x.information_gain-0.4*x.repetition))
        return Evaluation(next(k for k in PRECEDENCE if k in signals),tuple(k for k in PRECEDENCE if k in signals),investigate,repeat,contradiction)
