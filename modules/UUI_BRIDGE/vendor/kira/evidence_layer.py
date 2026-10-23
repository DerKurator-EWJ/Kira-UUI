"""Persistent provenance-aware evidence and non-destructive belief revision.

Evidence records are immutable. Claims are contextual hypotheses, not facts.
Conflicts open review cases; a contradiction never silently replaces a belief.
"""
import json
import sqlite3
import uuid
from datetime import datetime, timezone

SOURCES = {'observation', 'action_result', 'external_claim', 'inference', 'premise'}


def _now():
    return datetime.now(timezone.utc).isoformat()


class EvidenceLayer:
    def __init__(self, db):
        self.db = db
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS kira_evidence (
                evidence_id TEXT PRIMARY KEY, event_id TEXT UNIQUE NOT NULL,
                subject TEXT NOT NULL, predicate TEXT NOT NULL, value_json TEXT NOT NULL,
                context_json TEXT NOT NULL, source_type TEXT NOT NULL,
                source_ref TEXT NOT NULL, confidence REAL NOT NULL,
                recorded_at TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS kira_evidence_lookup
                ON kira_evidence(subject,predicate);
            CREATE TABLE IF NOT EXISTS kira_beliefs (
                belief_id TEXT PRIMARY KEY, evidence_id TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL CHECK(status IN ('active','superseded','rejected')),
                created_at TEXT NOT NULL, FOREIGN KEY(evidence_id) REFERENCES kira_evidence(evidence_id));
            CREATE TABLE IF NOT EXISTS kira_reviews (
                review_id TEXT PRIMARY KEY, new_evidence_id TEXT NOT NULL,
                old_evidence_id TEXT NOT NULL, status TEXT NOT NULL
                    CHECK(status IN ('open','resolved')),
                resolution TEXT, resolved_at TEXT,
                UNIQUE(new_evidence_id,old_evidence_id));
            CREATE INDEX IF NOT EXISTS kira_reviews_status ON kira_reviews(status);
        ''')
        self.db.commit()

    def record(self, *, event_id, subject, predicate, value, context=None,
               source_type='observation', source_ref, confidence=1.0):
        if source_type not in SOURCES:
            raise ValueError('invalid source_type')
        if not all(isinstance(v,str) and v.strip() for v in (event_id,subject,predicate,source_ref)):
            raise ValueError('event_id, subject, predicate, source_ref required')
        if not isinstance(confidence,(int,float)) or not 0 <= confidence <= 1:
            raise ValueError('confidence must be in [0,1]')
        context = {} if context is None else context
        if not isinstance(context,dict):
            raise ValueError('context must be a dictionary')
        value_json=json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False)
        context_json=json.dumps(context,sort_keys=True,ensure_ascii=False,allow_nan=False)
        old=self.db.execute('SELECT evidence_id FROM kira_evidence WHERE event_id=?',(event_id,)).fetchone()
        if old:
            raise ValueError('duplicate evidence event_id')
        evidence_id=str(uuid.uuid4())
        with self.db:
            self.db.execute('INSERT INTO kira_evidence VALUES(?,?,?,?,?,?,?,?,?,?)',
                (evidence_id,event_id,subject,predicate,value_json,context_json,source_type,source_ref,float(confidence),_now()))
            self.db.execute('INSERT INTO kira_beliefs VALUES(?,?,?,?)',
                (str(uuid.uuid4()),evidence_id,'active',_now()))
            rows=self.db.execute('''SELECT e.evidence_id,e.value_json,e.context_json FROM kira_evidence e
                JOIN kira_beliefs b ON b.evidence_id=e.evidence_id
                WHERE e.subject=? AND e.predicate=? AND b.status='active' AND e.evidence_id<>?''',
                (subject,predicate,evidence_id)).fetchall()
            reviews=[]
            for previous,old_value,old_context in rows:
                # Context is part of the hypothesis. Different conditions need not conflict.
                if old_context == context_json and old_value != value_json:
                    rid=str(uuid.uuid4())
                    self.db.execute('INSERT INTO kira_reviews(review_id,new_evidence_id,old_evidence_id,status) VALUES(?,?,?,?)',
                        (rid,evidence_id,previous,'open'))
                    reviews.append(rid)
        return {'evidence_id':evidence_id,'review_ids':reviews,'needs_review':bool(reviews)}

    def pending_reviews(self):
        return [dict(review_id=r[0],new_evidence_id=r[1],old_evidence_id=r[2])
                for r in self.db.execute('SELECT review_id,new_evidence_id,old_evidence_id FROM kira_reviews WHERE status=? ORDER BY rowid',('open',))]

    def resolve(self, review_id, resolution):
        # Decisions require an explicit reason; they never erase evidence.
        choices={'retain_both','supersede_old','reject_new'}
        if resolution not in choices: raise ValueError('invalid resolution')
        with self.db:
            row=self.db.execute('SELECT new_evidence_id,old_evidence_id,status FROM kira_reviews WHERE review_id=?',(review_id,)).fetchone()
            if row is None or row[2]!='open': raise ValueError('review not open')
            new,old,_=row
            if resolution=='supersede_old':
                self.db.execute("UPDATE kira_beliefs SET status='superseded' WHERE evidence_id=?",(old,))
            elif resolution=='reject_new':
                self.db.execute("UPDATE kira_beliefs SET status='rejected' WHERE evidence_id=?",(new,))
            self.db.execute('UPDATE kira_reviews SET status=?,resolution=?,resolved_at=? WHERE review_id=?',
                            ('resolved',resolution,_now(),review_id))

    def evidence_for(self,subject,predicate):
        rows=self.db.execute('''SELECT e.event_id,e.value_json,e.context_json,e.source_type,e.source_ref,
            e.confidence,b.status FROM kira_evidence e JOIN kira_beliefs b ON b.evidence_id=e.evidence_id
            WHERE e.subject=? AND e.predicate=? ORDER BY e.rowid''',(subject,predicate)).fetchall()
        return [dict(event_id=r[0],value=json.loads(r[1]),context=json.loads(r[2]),
                     source_type=r[3],source_ref=r[4],confidence=r[5],status=r[6]) for r in rows]
