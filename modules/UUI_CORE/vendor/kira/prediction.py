"""Online variable-order transition counts; predictive, not causal."""
import json
from collections import Counter

def signature(tokens):
    return json.dumps(list(tokens),separators=(',',':'))

class TransitionPredictor:
    def __init__(self, db, stream, max_context=3):
        if max_context < 1: raise ValueError('max_context must be >=1')
        self.db=db;self.stream=stream;self.max_context=max_context
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS prediction_counts(stream TEXT NOT NULL, context_len INTEGER NOT NULL, context TEXT NOT NULL, next_token INTEGER NOT NULL, occurrences INTEGER NOT NULL, PRIMARY KEY(stream,context_len,context,next_token));
        CREATE TABLE IF NOT EXISTS prediction_log(stream TEXT NOT NULL, tick INTEGER NOT NULL, context_len INTEGER NOT NULL, context TEXT NOT NULL, predicted_token INTEGER, observed_token INTEGER NOT NULL, confidence REAL NOT NULL, correct INTEGER, PRIMARY KEY(stream,tick));
        ''')
    def distribution(self, history, min_samples=1):
        if min_samples < 1:raise ValueError('min_samples must be >=1')
        for n in range(min(len(history),self.max_context),0,-1):
            ctx=signature(history[-n:]);rows=self.db.execute('SELECT next_token,occurrences FROM prediction_counts WHERE stream=? AND context_len=? AND context=? ORDER BY occurrences DESC,next_token ASC',(self.stream,n,ctx)).fetchall()
            total=sum(row[1] for row in rows)
            if total>=min_samples:
                return {'context':list(history[-n:]),'samples':total,'predictions':[{'token':token,'count':count,'probability':count/total} for token,count in rows]}
        return {'context':[],'samples':0,'predictions':[]}
    def record(self, tick, history, observed):
        """Called inside caller's observation transaction, BEFORE learning this observation."""
        dist=self.distribution(history)
        top=dist['predictions'][0] if dist['predictions'] else None
        predicted=top['token'] if top else None
        confidence=top['probability'] if top else 0.0
        correct=int(predicted==observed) if top else None
        self.db.execute('INSERT INTO prediction_log VALUES(?,?,?,?,?,?,?,?)',(self.stream,tick,len(dist['context']),signature(dist['context']),predicted,observed,confidence,correct))
        for n in range(1,min(len(history),self.max_context)+1):
            self.db.execute('''INSERT INTO prediction_counts VALUES(?,?,?,?,1) ON CONFLICT(stream,context_len,context,next_token) DO UPDATE SET occurrences=occurrences+1''',(self.stream,n,signature(history[-n:]),observed))
        return {'predicted':predicted,'observed':observed,'confidence':confidence,'correct':None if correct is None else bool(correct),'context':dist['context']}
    def metrics(self):
        total,scored,correct=self.db.execute('SELECT COUNT(*),COUNT(correct),COALESCE(SUM(correct),0) FROM prediction_log WHERE stream=?',(self.stream,)).fetchone()
        return {'observations':total,'scored_predictions':scored,'correct_predictions':correct,'accuracy':correct/scored if scored else None,'cold_start':total-scored}
    def recent(self,limit=10):
        if not 1<=limit<=1000:raise ValueError('limit must be 1..1000')
        rows=self.db.execute('SELECT tick,context,predicted_token,observed_token,confidence,correct FROM prediction_log WHERE stream=? ORDER BY tick DESC LIMIT ?',(self.stream,limit)).fetchall()
        return [{'tick':t,'context':json.loads(c),'predicted':p,'observed':o,'confidence':cf,'correct':None if ok is None else bool(ok)} for t,c,p,o,cf,ok in rows]
