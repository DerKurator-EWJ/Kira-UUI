"""Numeric Stream v1.0 -> decision matrix adapter with durable evaluation journal.
Prediction error is computed ONLY if a numeric prediction exists. Contextual metrics
must come from the caller; they are never fabricated from a numeric value.
"""
import json
from fractions import Fraction
from numeric_stream_v10 import NumericStreamV10
from numeric_stream_v07 import enc, dec
from decision_matrix import DecisionMatrix, EvaluationInput

class EvaluatedNumericStream:
    def __init__(self,path=':memory:',max_period=16,min_savings=1):
        self.core=NumericStreamV10(path,max_period,min_savings)
        self.matrix=DecisionMatrix()
        self.core.db.execute("""CREATE TABLE IF NOT EXISTS kira_evaluations (
            event_id TEXT PRIMARY KEY, series TEXT NOT NULL, observed TEXT NOT NULL,
            predicted TEXT, evaluation_json TEXT NOT NULL)""")
        self.core.db.commit()

    def observe_and_evaluate(self,event_id,series,value,*,success,novelty,relevance,
                             information_gain,valence,repetition=0.,contradiction_scope=0.,
                             suspicious_recurrence=0.,uncertainty=0.,
                             prediction_scale=1, expectation_confidence=None):
        if not event_id or not series:raise ValueError('event_id and series required')
        # Idempotence: duplicate event does not append a second observation.
        if self.core.db.execute('SELECT 1 FROM kira_evaluations WHERE event_id=?',(event_id,)).fetchone():
            raise ValueError('duplicate event_id')
        scale=Fraction(prediction_scale)
        if scale<=0:raise ValueError('prediction_scale must be positive')
        observed=dec(enc(value))
        predicted=self.core.predict(series)
        if predicted is None:
            confidence=0.0
            surprise=0.0
        else:
            confidence=1.0 if expectation_confidence is None else expectation_confidence
            surprise=float(min(Fraction(1),abs(observed-predicted)/scale))
        inp=EvaluationInput(confidence,surprise,success,novelty,relevance,information_gain,
                            valence,repetition,contradiction_scope,suspicious_recurrence,uncertainty)
        result=self.matrix.evaluate(inp)
        self.core.observe(series,observed)
        payload={'primary':result.primary,'utterance':result.utterance,'signals':result.signals,
                 'investigate_priority':result.investigate_priority,
                 'repeat_priority':result.repeat_priority,
                 'evidence_review_required':result.evidence_review_required,
                 'surprise':surprise,'expectation_confidence':confidence}
        with self.core.db:
            self.core.db.execute('INSERT INTO kira_evaluations VALUES(?,?,?,?,?)',
                (event_id,series,enc(observed),None if predicted is None else enc(predicted),json.dumps(payload)))
        return payload

    def history(self,series):
        return [dict(event_id=e,observed=o,predicted=p,**json.loads(v)) for e,o,p,v in
            self.core.db.execute('SELECT event_id,observed,predicted,evaluation_json FROM kira_evaluations WHERE series=? ORDER BY rowid',(series,))]

    def close(self):self.core.close()
