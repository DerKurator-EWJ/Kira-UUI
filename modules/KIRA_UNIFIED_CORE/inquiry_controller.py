"""Persistent, non-destructive inquiry queue for unresolved relations and evidence conflicts.

Questions are templates for investigation, not claims of autonomous reasoning.
"""
import json
import sqlite3
import uuid
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


class InquiryController:
    def __init__(self, evidence_layer):
        self.evidence = evidence_layer
        self.db = evidence_layer.db
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS kira_inquiries (
            inquiry_id TEXT PRIMARY KEY, dedupe_key TEXT UNIQUE NOT NULL,
            kind TEXT NOT NULL, subject TEXT NOT NULL, predicate TEXT NOT NULL,
            question TEXT NOT NULL, evidence_refs_json TEXT NOT NULL,
            priority REAL NOT NULL, status TEXT NOT NULL CHECK(status IN ('open','answered','dismissed')),
            answer_ref TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS kira_inquiries_queue ON kira_inquiries(status,priority);
        ''')
        self.db.commit()

    def _enqueue(self, key, kind, subject, predicate, question, refs=(), priority=0.5):
        if not isinstance(priority,(int,float)) or not 0 <= priority <= 1:
            raise ValueError('priority must be in [0,1]')
        if not subject or not predicate or not question:
            raise ValueError('subject, predicate and question required')
        with self.db:
            existing=self.db.execute('SELECT inquiry_id FROM kira_inquiries WHERE dedupe_key=?',(key,)).fetchone()
            if existing:
                return existing[0]
            ident=str(uuid.uuid4())
            self.db.execute('INSERT INTO kira_inquiries VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                (ident,key,kind,subject,predicate,question,json.dumps(list(refs)),float(priority),'open',None,_now(),_now()))
            return ident

    def ask_about(self, subject, predicate, *, priority=0.45, context=None):
        """Generate curiosity from an ordinary unknown relation, without prediction error."""
        context={} if context is None else context
        if not isinstance(context,dict): raise ValueError('context must be dict')
        key=json.dumps(['unknown',subject,predicate,context],sort_keys=True,ensure_ascii=False)
        return self._enqueue(key,'unknown',subject,predicate,
            f'Welche Bedeutung oder Funktion hat {predicate} bei {subject}?',priority=priority)

    def sync_reviews(self):
        """Make one open inquiry per open review; keep evidence immutable."""
        created=[]
        for review in self.evidence.pending_reviews():
            pair=self.db.execute('''SELECT subject,predicate FROM kira_evidence WHERE evidence_id=?''',
                                 (review['new_evidence_id'],)).fetchone()
            if pair is None: continue
            subject,predicate=pair
            key='review:'+review['review_id']
            created.append(self._enqueue(key,'contradiction',subject,predicate,
                f'Warum gibt es widersprüchliche Beobachtungen zu {subject} / {predicate}? '
                'Haben sich Kontext, Objekt oder Messung verändert?',
                (review['old_evidence_id'],review['new_evidence_id']),0.9))
        return created

    def next_questions(self,limit=10):
        if not isinstance(limit,int) or limit<1 or limit>1000:raise ValueError('invalid limit')
        rows=self.db.execute('''SELECT inquiry_id,kind,subject,predicate,question,
            evidence_refs_json,priority,status FROM kira_inquiries WHERE status='open'
            ORDER BY priority DESC,created_at ASC,inquiry_id ASC LIMIT ?''',(limit,)).fetchall()
        return [dict(inquiry_id=r[0],kind=r[1],subject=r[2],predicate=r[3],question=r[4],
                     evidence_refs=json.loads(r[5]),priority=r[6],status=r[7]) for r in rows]

    def answer(self,inquiry_id,answer_ref):
        """Record an answer reference; does NOT automatically settle contradictory evidence."""
        if not isinstance(answer_ref,str) or not answer_ref.strip():
            raise ValueError('nonempty answer_ref required')
        with self.db:
            row=self.db.execute('SELECT status FROM kira_inquiries WHERE inquiry_id=?',(inquiry_id,)).fetchone()
            if not row or row[0]!='open':raise ValueError('inquiry not open')
            self.db.execute('UPDATE kira_inquiries SET status=?,answer_ref=?,updated_at=? WHERE inquiry_id=?',
                            ('answered',answer_ref,_now(),inquiry_id))

    def dismiss(self,inquiry_id):
        with self.db:
            row=self.db.execute('SELECT status FROM kira_inquiries WHERE inquiry_id=?',(inquiry_id,)).fetchone()
            if not row or row[0]!='open':raise ValueError('inquiry not open')
            self.db.execute('UPDATE kira_inquiries SET status=?,updated_at=? WHERE inquiry_id=?',
                            ('dismissed',_now(),inquiry_id))
