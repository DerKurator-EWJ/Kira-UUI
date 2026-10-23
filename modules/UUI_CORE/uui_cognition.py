"""UUI 0.4 cognitive standard library. Unimplemented operations fail explicitly."""
from fractions import Fraction
from dataclasses import dataclass
import uuid
import json
from uui_revision import install, propose, apply, history

FAMILIES = {
 'wahrnehmung': ('guck','hoer','spuer'),
 'identitaet': ('wer_bist_du','ist_das_noch','unterscheide'),
 'relationen': ('was_haengt_dran','verknuepf','trenn'),
 'gedaechtnis': ('merk_dir','erinner_dich','vergiss_nicht'),
 'erkenntnis': ('was_weisst_du','woher_weisst_du','bist_du_sicher'),
 'staunen': ('was_ist_das','warum','warum_nicht'),
 'vorhersage': ('was_waere_wenn','was_kommt','wie_sicher'),
 'entscheidung': ('mach_sinn','lohnt_sich','lass_stecken'),
 'handlung': ('tu_es','probier_mal','hoer_auf'),
 'erfahrung': ('hat_geklappt','war_nix','lern_draus'),
 'makros': ('fass_zusammen','erkenn_wieder','entfalte'),
 'selbstpruefung': ('pruef_dich','such_widerspruch','was_fehlt'),
 'system': ('wer_darf','was_kostet','stopp'),
}
ALL = {name: family for family, names in FAMILIES.items() for name in names}
REVISION_OPS = {name:"revision" for name in ("pruef_revision","entscheide_revision","revision_verlauf")}
assert len(ALL)==39

# Operations return JSON-serializable strings where UUI 0.3 lacks maps/lists.
# No access to cameras, microphones, robots or network is implied.
def invoke(name, args, runtime):
    def arity(n):
        if len(args)!=n: raise ValueError(f'{name} expects {n} arguments')
    def string(v):
        if not isinstance(v,str):raise ValueError(f'{name} expects a string')
        return v
    def number(v):
        if type(v) is not Fraction:raise ValueError(f'{name} expects a number')
        return v
    # Evidence functions return JSON strings because UUI 0.4 has no record/list type.
    # All observations are explicitly attributed; inquiries do not invent answers.
    if name=='pruef_revision':
        arity(1); return json.dumps(propose(runtime.bridge.evidence,string(args[0])),ensure_ascii=False,sort_keys=True)
    if name=='entscheide_revision':
        arity(3); return json.dumps(apply(runtime.bridge.evidence,*map(string,args)),ensure_ascii=False,sort_keys=True)
    if name=='revision_verlauf':
        arity(1); return json.dumps(history(runtime.bridge.evidence,string(args[0])),ensure_ascii=False,sort_keys=True)
    if name=='was_weisst_du':
        arity(2); subject,predicate=map(string,args)
        records=runtime.bridge.evidence.evidence_for(subject,predicate)
        return json.dumps({'subject':subject,'predicate':predicate,'evidence':records,
                           'known':bool(records)},ensure_ascii=False,sort_keys=True)
    if name=='woher_weisst_du':
        arity(2); subject,predicate=map(string,args)
        records=runtime.bridge.evidence.evidence_for(subject,predicate)
        return json.dumps({'subject':subject,'predicate':predicate,
                           'sources':[{'event_id':r['event_id'],'source_type':r['source_type'],
                                       'source_ref':r['source_ref'],'confidence':r['confidence'],
                                       'status':r['status']} for r in records]},ensure_ascii=False,sort_keys=True)
    if name=='warum_nicht':
        if len(args)==1:
            return 'Welche Alternative oder Gegenbegruendung gibt es zu: '+str(args[0])
        arity(2); subject,predicate=map(string,args)
        ident=runtime.bridge.ask_about(subject,predicate)
        return json.dumps({'inquiry_id':ident,'subject':subject,'predicate':predicate,
                           'status':'open','kind':'unknown'},ensure_ascii=False,sort_keys=True)
    if name=='lern_draus':
        # A supplied observation is recorded as evidence, not automatically promoted
        # to truth. Contradictions remain open for human/system review.
        if len(args) not in (5,6,7):
            raise ValueError('lern_draus expects 5, 6 or 7 arguments')
        subject,predicate,value,source_ref,event_id=args[:5]
        for x in (subject,predicate,value,source_ref,event_id):string(x)
        source_type = string(args[5]) if len(args)>=6 else 'external_claim'
        # Explicit provenance is mandatory for an observation; caller-supplied
        # source labels are not independently verified by the runtime.
        if source_type not in ('observation','action_result','external_claim','inference','premise'):
            raise ValueError('invalid evidence source_type')
        confidence=1.0
        if len(args)==7:
            c=number(args[6])
            if not 0<=c<=1:raise ValueError('confidence must be between 0 and 1')
            confidence=float(c)
        result=runtime.bridge.evidence.record(event_id=event_id,subject=subject,
            predicate=predicate,value=value,source_type=source_type,
            source_ref=source_ref,confidence=confidence)
        runtime.bridge.inquiry.sync_reviews()
        return json.dumps({'recorded':True,'source_type':source_type,'confidence':confidence,**result},ensure_ascii=False,sort_keys=True)
    if name=='was_waere_wenn':
        # Isolated counterfactual arithmetic; does not mutate runtime or evidence.
        arity(3); initial=number(args[0]);delta=number(args[1]);operation=string(args[2])
        if operation not in ('plus','minus','mal','geteilt'):
            raise ValueError('was_waere_wenn operation must be plus/minus/mal/geteilt')
        if operation=='geteilt' and delta==0:raise ValueError('Division by zero')
        result={'plus':lambda:initial+delta,'minus':lambda:initial-delta,
                'mal':lambda:initial*delta,'geteilt':lambda:initial/delta}[operation]()
        return result
    if name=='such_widerspruch':
        arity(0)
        return json.dumps({'reviews':runtime.bridge.evidence.pending_reviews()},ensure_ascii=False,sort_keys=True)
    if name=='pruef_dich':
        arity(2); subject,predicate=map(string,args)
        evidence=runtime.bridge.evidence.evidence_for(subject,predicate)
        active=[r for r in evidence if r['status']=='active']
        contexts={json.dumps(r['context'],sort_keys=True,ensure_ascii=False) for r in active}
        distinct={(json.dumps(r['value'],sort_keys=True,ensure_ascii=False),json.dumps(r['context'],sort_keys=True,ensure_ascii=False)) for r in active}
        reviews=runtime.bridge.evidence.pending_reviews()
        return json.dumps({'subject':subject,'predicate':predicate,'known':bool(evidence),
          'evidence_count':len(evidence),'active_count':len(active),
          'open_reviews':len(reviews),'conflict_possible':len(distinct)>len(contexts),
          'verdict':'requires_review' if len(distinct)>len(contexts) else 'not_verified'},
          ensure_ascii=False,sort_keys=True)
    if name=='was_ist_das':
        arity(1);return 'offen: '+str(args[0])
    if name=='warum':
        arity(1);return ('Welche Begruendung fehlt fuer: ' if name=='warum' else 'Welche Alternative oder Gegenbegruendung gibt es zu: ')+str(args[0])
    if name=='wie_sicher':
        arity(1);x=number(args[0]);return x>=0 and x<=1
    if name=='ist_das_noch':
        arity(2);return type(args[0])==type(args[1]) and args[0]==args[1] # exact equality, not tracking
    if name=='unterscheide':
        arity(2);return not(type(args[0])==type(args[1]) and args[0]==args[1])
    if name=='was_kostet':
        arity(0);return Fraction(runtime.steps)
    if name=='merk_dir':
        arity(2);value=number(args[0]);source=string(args[1]);return json.dumps(runtime.bridge.numeric_observation(str(value),source=source),ensure_ascii=False)
    if name=='erinner_dich':
        arity(1);i=number(args[0]);
        if i.denominator!=1 or i<=0:raise ValueError('numeric observation ID must be a positive integer')
        v=runtime.bridge.numeric.value(int(i));return v
    if name=='wer_darf':
        arity(1);return False # default-deny; no grants in prototype
    if name=='stopp':
        arity(0);raise ValueError('Program stopped by stopp()')
    if name=='hat_geklappt':
        arity(1)
        if type(args[0]) is not bool:raise ValueError('hat_geklappt expects a boolean')
        return args[0]
    if name=='war_nix':
        arity(1)
        if type(args[0]) is not bool:raise ValueError('war_nix expects a boolean')
        return not args[0]
    if name=='was_fehlt':
        arity(1);return 'offene Frage: '+str(args[0])
    if name=='bist_du_sicher':
        arity(1);return False # cannot certify evidence
    raise NotImplementedError(f'{name} is registered but not implemented; requires a dedicated Kira capability')
