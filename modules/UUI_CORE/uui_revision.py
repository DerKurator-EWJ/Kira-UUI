"""Conservative evidence review: proposals are not automatic truth judgments."""
import json
from datetime import datetime, timezone


def install(evidence):
    evidence.db.executescript('''
    CREATE TABLE IF NOT EXISTS uui_revision_audit (
      audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
      review_id TEXT NOT NULL, decision TEXT NOT NULL,
      reason TEXT NOT NULL, proposed_at TEXT NOT NULL,
      applied INTEGER NOT NULL DEFAULT 0);
    ''')
    evidence.db.commit()


def propose(evidence, review_id):
    row=evidence.db.execute('''SELECT r.status, n.value_json, o.value_json,
      n.context_json,o.context_json,n.source_type,o.source_type,
      n.confidence,o.confidence,n.source_ref,o.source_ref
      FROM kira_reviews r JOIN kira_evidence n ON n.evidence_id=r.new_evidence_id
      JOIN kira_evidence o ON o.evidence_id=r.old_evidence_id
      WHERE r.review_id=?''',(review_id,)).fetchone()
    if row is None: raise ValueError('unknown review')
    status,new,old,nctx,octx,nt,ot,nc,oc,ns,os=row
    # Confidence is user supplied, not calibrated. No automatic resolution.
    reasons=['Conflicting claims require independent verification.',
             'Confidence values are caller-supplied and not calibrated.']
    if nctx!=octx: reasons.append('Different contexts may explain the disagreement.')
    if ns==os: reasons.append('Sources may be correlated; independence is not established.')
    return {'review_id':review_id,'status':status,'suggested':'retain_both',
            'automatic_resolution':False,'new_value':json.loads(new),
            'old_value':json.loads(old),'new_source_type':nt,'old_source_type':ot,
            'new_confidence':nc,'old_confidence':oc,'reasons':reasons,
            'options':['retain_both','supersede_old','reject_new']}


def apply(evidence, review_id, decision, reason):
    if decision not in ('retain_both','supersede_old','reject_new'):
        raise ValueError('invalid revision decision')
    if not isinstance(reason,str) or len(reason.strip())<8:
        raise ValueError('explicit reason of at least 8 characters required')
    install(evidence)
    with evidence.db:
        evidence.resolve(review_id,decision)
        evidence.db.execute('''INSERT INTO uui_revision_audit
            (review_id,decision,reason,proposed_at,applied) VALUES (?,?,?,?,1)''',
            (review_id,decision,reason.strip(),datetime.now(timezone.utc).isoformat()))
    return {'review_id':review_id,'decision':decision,'applied':True,'reason':reason.strip()}


def history(evidence, review_id):
    install(evidence)
    rows=evidence.db.execute('''SELECT decision,reason,proposed_at,applied
        FROM uui_revision_audit WHERE review_id=? ORDER BY audit_id''',(review_id,)).fetchall()
    return [{'decision':r[0],'reason':r[1],'at':r[2],'applied':bool(r[3])} for r in rows]
