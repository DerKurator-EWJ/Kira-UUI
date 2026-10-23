"""UUI 0.4 cognitive standard library. Unimplemented operations fail explicitly."""
from fractions import Fraction
from dataclasses import dataclass
import json

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
    if name=='was_ist_das':
        arity(1);return 'offen: '+str(args[0])
    if name=='warum' or name=='warum_nicht':
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
