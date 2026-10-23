"""Conservative inquiry-to-experiment planner.

Actions are registered capabilities, never invented executable commands. Planning
is separate from execution; observed outcomes are recorded as evidence only
when the caller explicitly supplies a verified result.
"""
import json
import math
import uuid
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


class ExperimentPlanner:
    def __init__(self, inquiry_controller):
        self.inquiry = inquiry_controller
        self.evidence = inquiry_controller.evidence
        self.db = inquiry_controller.db
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS kira_capabilities (
            capability_id TEXT PRIMARY KEY, predicate TEXT NOT NULL,
            description TEXT NOT NULL, risk REAL NOT NULL,
            cost REAL NOT NULL, enabled INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS kira_experiments (
            experiment_id TEXT PRIMARY KEY, inquiry_id TEXT NOT NULL,
            capability_id TEXT NOT NULL, subject TEXT NOT NULL,
            predicted_value_json TEXT, expected_information REAL NOT NULL,
            score REAL NOT NULL, status TEXT NOT NULL
                CHECK(status IN ('proposed','approved','completed','cancelled')),
            result_evidence_id TEXT, created_at TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS kira_experiments_inquiry
            ON kira_experiments(inquiry_id,status);
        ''')
        self.db.commit()

    def register_capability(self, capability_id, predicate, description, *, risk=0.0, cost=0.0, enabled=True):
        if not all(isinstance(v,str) and v.strip() for v in (capability_id,predicate,description)):
            raise ValueError('capability id, predicate and description required')
        for v in (risk,cost):
            if not isinstance(v,(int,float)) or not math.isfinite(v) or not 0 <= v <= 1:
                raise ValueError('risk and cost must be finite in [0,1]')
        with self.db:
            self.db.execute('''INSERT INTO kira_capabilities VALUES(?,?,?,?,?,?)
                ON CONFLICT(capability_id) DO UPDATE SET predicate=excluded.predicate,
                description=excluded.description,risk=excluded.risk,cost=excluded.cost,
                enabled=excluded.enabled''',
                (capability_id,predicate,description,float(risk),float(cost),int(bool(enabled))))

    def propose(self, inquiry_id, *, limit=5):
        if not isinstance(limit,int) or limit < 1 or limit > 100: raise ValueError('invalid limit')
        q=self.db.execute('''SELECT subject,predicate,kind,priority,status
            FROM kira_inquiries WHERE inquiry_id=?''',(inquiry_id,)).fetchone()
        if not q or q[4]!='open': raise ValueError('inquiry not open')
        subject,predicate,kind,priority,_=q
        capabilities=self.db.execute('''SELECT capability_id,description,risk,cost
            FROM kira_capabilities WHERE predicate=? AND enabled=1 ORDER BY capability_id''',(predicate,)).fetchall()
        prior=self.evidence.evidence_for(subject,predicate)
        # This is a transparent heuristic, not a learned world model.
        info=1.0 if not prior else (0.85 if kind=='contradiction' else 0.5)
        options=[]
        for cap_id,description,risk,cost in capabilities:
            score=round(max(0.0, priority*info*(1-risk)*(1-0.5*cost)),6)
            options.append(dict(capability_id=cap_id,description=description,
                                expected_information=info,score=score,risk=risk,cost=cost))
        options.sort(key=lambda x:(-x['score'],x['capability_id']))
        proposed=[]
        with self.db:
            for option in options[:limit]:
                cap_id=option['capability_id']
                row=self.db.execute('''SELECT experiment_id,status FROM kira_experiments
                    WHERE inquiry_id=? AND capability_id=? AND status IN ('proposed','approved')
                    ORDER BY created_at DESC LIMIT 1''',(inquiry_id,cap_id)).fetchone()
                ident=row[0] if row else str(uuid.uuid4())
                if not row:
                    self.db.execute('''INSERT INTO kira_experiments VALUES(?,?,?,?,?,?,?,?,?,?)''',
                        (ident,inquiry_id,cap_id,subject,None,info,option['score'],'proposed',None,_now()))
                proposed.append(dict(experiment_id=ident,inquiry_id=inquiry_id,subject=subject,**option))
        return proposed

    def approve(self, experiment_id):
        with self.db:
            row=self.db.execute('SELECT status FROM kira_experiments WHERE experiment_id=?',(experiment_id,)).fetchone()
            if not row or row[0]!='proposed': raise ValueError('experiment not proposed')
            self.db.execute("UPDATE kira_experiments SET status='approved' WHERE experiment_id=?",(experiment_id,))

    def complete(self, experiment_id, *, event_id, value, source_ref, context=None, confidence=1.0):
        """Record a caller-verified observation; never physically execute actions."""
        row=self.db.execute('''SELECT x.status,x.subject,c.predicate,x.inquiry_id
            FROM kira_experiments x JOIN kira_capabilities c USING(capability_id)
            WHERE x.experiment_id=?''',(experiment_id,)).fetchone()
        if not row or row[0]!='approved': raise ValueError('experiment not approved')
        _,subject,predicate,inquiry_id=row
        result=self.evidence.record(event_id=event_id,subject=subject,predicate=predicate,
            value=value,context=context,source_type='action_result',source_ref=source_ref,
            confidence=confidence)
        with self.db:
            self.db.execute('''UPDATE kira_experiments SET status='completed',result_evidence_id=?
                WHERE experiment_id=?''',(result['evidence_id'],experiment_id))
        self.inquiry.sync_reviews()
        return result

    def cancel(self, experiment_id):
        with self.db:
            row=self.db.execute('SELECT status FROM kira_experiments WHERE experiment_id=?',(experiment_id,)).fetchone()
            if not row or row[0] not in ('proposed','approved'): raise ValueError('experiment not cancellable')
            self.db.execute("UPDATE kira_experiments SET status='cancelled' WHERE experiment_id=?",(experiment_id,))
