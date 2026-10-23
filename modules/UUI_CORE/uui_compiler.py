"""UUIX 0.3 portable JSON AST container; NOT machine code or stack bytecode."""
import hashlib
import json
from uui_lang import Parser, tokenize, UUIError

FORMAT='uui.uuix.ast.v1'

def compile_source(source):
    tokens=tokenize(source)
    if len(tokens)>20000:raise UUIError('Token budget exceeded')
    program=Parser(source,tokens).program()
    if len(program)>1000:raise UUIError('Statement budget exceeded')
    body=json.dumps(program,ensure_ascii=False,separators=(',',':'))
    return {'format':FORMAT,'sha256':hashlib.sha256(body.encode('utf-8')).hexdigest(),'program':program}

def serialize(compiled):
    return json.dumps(compiled,ensure_ascii=False,separators=(',',':')).encode('utf-8')

def deserialize(blob):
    if len(blob)>2_000_000:raise UUIError('UUIX file too large')
    try:obj=json.loads(blob)
    except (ValueError,UnicodeDecodeError) as e:raise UUIError('Invalid UUIX JSON') from e
    if not isinstance(obj,dict) or obj.get('format')!=FORMAT or not isinstance(obj.get('program'),list):
        raise UUIError('Unsupported UUIX format')
    body=json.dumps(obj['program'],ensure_ascii=False,separators=(',',':'))
    if hashlib.sha256(body.encode('utf-8')).hexdigest()!=obj.get('sha256'):
        raise UUIError('UUIX integrity check failed')
    validate_ast(obj['program'])
    return obj['program']

def validate_ast(program):
    # Restrict the JSON IR to the parser's supported statement/expression shapes.
    def expr(n,depth=0):
        if depth>100 or not isinstance(n,list) or not n:raise UUIError('Invalid expression IR')
        k=n[0]
        if k=='num' and len(n)==2 and isinstance(n[1],str):
            from fractions import Fraction
            try:Fraction(n[1])
            except ValueError as e:raise UUIError('Invalid numeric IR') from e
        elif k=='str' and len(n)==2 and isinstance(n[1],str):pass
        elif k=='bool' and len(n)==2 and type(n[1]) is bool:pass
        elif k=='var' and len(n)==2 and isinstance(n[1],str):pass
        elif k=='unary' and len(n)==3 and n[1] in ('-','!','not'):expr(n[2],depth+1)
        elif k=='bin' and len(n)==4 and n[1] in ('+','-','*','/','==','!=','<','>','<=','>=','and','or'):
            expr(n[2],depth+1);expr(n[3],depth+1)
        elif k=='call' and len(n)==3 and isinstance(n[1],str) and isinstance(n[2],list):
            for a in n[2]:expr(a,depth+1)
        else:raise UUIError('Unsupported expression IR')
    def statements(seq,depth=0):
        if depth>100 or not isinstance(seq,list) or len(seq)>1000:raise UUIError('Invalid statement IR')
        for n in seq:
            if not isinstance(n,list) or not n:raise UUIError('Invalid statement IR')
            k=n[0]
            if k in ('let','set') and len(n)==3 and isinstance(n[1],str):expr(n[2])
            elif k in ('print','return') and len(n)==2:expr(n[1])
            elif k in ('observe','resolve') and len(n)==3:expr(n[1]);expr(n[2])
            elif k=='if' and len(n)==4:expr(n[1]);statements(n[2],depth+1);statements(n[3],depth+1)
            elif k=='while' and len(n)==3:expr(n[1]);statements(n[2],depth+1)
            elif k=='fn' and len(n)==4 and isinstance(n[1],str) and isinstance(n[2],list) and all(isinstance(x,str) for x in n[2]):statements(n[3],depth+1)
            else:raise UUIError('Unsupported statement IR')
    statements(program)
