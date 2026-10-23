"""Minimal task-based dispatch: retain specialized token predictor and relational graph learner.
One SQLite connection, one persistent DB, no bridge or duplicated graph snapshots.
"""
from binary_learning import BinaryLearningRuntime
from prediction import TransitionPredictor
from numeric_core import NumericCore

class EfficientCore:
    def __init__(self, path=':memory:', cache_budget=65536, max_context=1):
        self.rrm=BinaryLearningRuntime(path,cache_budget)
        self.numeric=NumericCore(path)
        self.max_context=max_context
        self._token_predictors={}
        self._token_state={}

    def _predictor(self,stream):
        if stream not in self._token_predictors:
            self._token_predictors[stream]=TransitionPredictor(self.rrm.db,stream,self.max_context)
        return self._token_predictors[stream]

    def observe(self,stream,value):
        if type(value) is int:
            p=self._predictor('token:'+stream)
            db=self.rrm.db
            key='token:'+stream
            if key not in self._token_state:
                rows=db.execute('SELECT tick,observed_token FROM prediction_log WHERE stream=? ORDER BY tick DESC LIMIT ?',(key,self.max_context)).fetchall()
                last=db.execute('SELECT MAX(tick) FROM prediction_log WHERE stream=?',(key,)).fetchone()[0]
                self._token_state[key]=[0 if last is None else last+1,[r[1] for r in reversed(rows)]]
            tick,history=self._token_state[key]
            with db:
                result=p.record(tick,history,value)
            self._token_state[key]=[tick+1,(history+[value])[-self.max_context:]]
            return {'route':'token','tick':tick,'evaluation':result}
        if isinstance(value,dict) and 'nodes' in value and 'edges' in value:
            return {'route':'graph','position':self.rrm.observe('graph:'+stream,value)}
        raise TypeError('expected integer token or graph {nodes, edges}; ambiguous inputs rejected')

    def predict(self,stream,value,**kwargs):
        if type(value) is int:
            if kwargs: raise TypeError('token route does not accept graph prediction options')
            return {'route':'token',**self._predictor('token:'+stream).distribution([value])}
        if isinstance(value,dict) and 'nodes' in value and 'edges' in value:
            return {'route':'graph',**self.rrm.predict('graph:'+stream,value,**kwargs)}
        raise TypeError('expected integer token or graph {nodes, edges}')

    def observe_numeric(self, value, identity=None, source="observed"):
        return self.numeric.observe(value, identity, source)

    def query_numeric(self, value, tolerance):
        return self.numeric.range(value, tolerance)

    def close(self):
        self.numeric.close()
        self.rrm.close()
