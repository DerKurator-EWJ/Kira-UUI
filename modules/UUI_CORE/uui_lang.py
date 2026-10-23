"""UUI Core 0.2: bounded interpreter; no Python eval/exec or arbitrary host access."""
from dataclasses import dataclass
from fractions import Fraction
import json
import re
from uui_cognition import ALL as CORE_OPS, REVISION_OPS, RELATION_OPS, PATTERN_OPS, invoke as invoke_cognitive
COGNITIVE_OPS = CORE_OPS | REVISION_OPS | RELATION_OPS | PATTERN_OPS

class UUIError(Exception): pass

@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    position: int

TOKEN = re.compile(r'''(?P<SPACE>[ \t\r]+)|(?P<COMMENT>\#[^\n]*)|(?P<NEWLINE>\n)|(?P<NUMBER>\d+(?:\.\d+)?)|(?P<STRING>"(?:\\["\\nrt]|[^"\\])*"|'(?:\\['\\nrt]|[^'\\])*')|(?P<NAME>[A-Za-z_][A-Za-z_0-9]*)|(?P<SYM>==|!=|<=|>=|[{}=()+*/;,<>!\-])|(?P<INVALID>.)''')
ALIASES = {'was_brauchst_du':'if','geht_das':'else','hier':'print','es_sei':'let',
           'so_nicht':'set','mach':'fn','her_damit':'return','solange':'while',
           'pass_uff':'observe'}
KEYWORDS={'set','let','print','observe','resolve','as','if','else','while','fn','return','true','false','and','or','not'} | set(ALIASES)


def tokenize(source):
    out=[]
    for m in TOKEN.finditer(source):
        kind=m.lastgroup
        if kind in ('SPACE','COMMENT'): continue
        if kind=='INVALID': raise UUIError(f'Unexpected character {m.group()!r} at {m.start()}')
        out.append(Token(kind,ALIASES.get(m.group(),m.group()) if kind=='NAME' else m.group(),m.start()))
    out.append(Token('EOF','',len(source)))
    return out

class Parser:
    def __init__(self,source,tokens=None): self.tokens=tokens if tokens is not None else tokenize(source);self.i=0
    def peek(self):return self.tokens[self.i]
    def take(self):t=self.peek();self.i+=1;return t
    def accept(self,s):
        if self.peek().value==s:self.take();return True
        return False
    def expect(self,s):
        if not self.accept(s):raise UUIError(f'Expected {s!r} at {self.peek().position}')
    def name(self):
        t=self.take()
        if t.kind!='NAME' or t.value in KEYWORDS:raise UUIError(f'Expected nonreserved identifier at {t.position}')
        return t.value
    def separators(self):
        count=0
        while self.peek().kind=='NEWLINE' or self.peek().value==';':self.take();count+=1
        return count
    def program(self,terminator='EOF'):
        out=[];self.separators()
        while self.peek().kind!=terminator and self.peek().value!=terminator:
            if self.peek().kind=='EOF':raise UUIError('Unclosed block')
            out.append(self.statement())
            if self.peek().kind!=terminator and self.peek().value!=terminator and not self.separators():
                raise UUIError(f'Expected newline or semicolon at {self.peek().position}')
            self.separators()
        return out
    def block(self):
        self.expect('{');body=self.program('}');self.expect('}');return body
    def statement(self):
        t=self.take()
        if t.kind!='NAME':raise UUIError(f'Unknown statement {t.value!r} at {t.position}')
        if t.value=='let':
            name=self.name();self.expect('=');return ('let',name,self.expr())
        if t.value=='set':
            name=self.name();self.expect('=');return ('set',name,self.expr())
        if t.value=='print':return ('print',self.expr())
        if t.value=='observe':
            val=self.expr();self.expect('as');return ('observe',val,self.expr())
        if t.value=='resolve':
            val=self.expr();self.expect(',');return ('resolve',val,self.expr())
        if t.value in ('if','while'):
            cond=self.expr();body=self.block()
            if t.value=='while':return ('while',cond,body)
            save=self.i
            self.separators()
            if self.accept('else'):other=self.block()
            else:self.i=save;other=[]
            return ('if',cond,body,other)
        if t.value=='fn':
            name=self.name();self.expect('(');params=[]
            if not self.accept(')'):
                while True:
                    p=self.name()
                    if p in params:raise UUIError('Duplicate parameter')
                    params.append(p)
                    if self.accept(')'):break
                    self.expect(',')
            return ('fn',name,params,self.block())
        if t.value=='return':return ('return',self.expr())
        raise UUIError(f'Unknown statement {t.value!r} at {t.position}')
    def expr(self,min_prec=0):
        t=self.take()
        if t.value in ('-','!','not'):left=('unary',t.value,self.expr(60))
        elif t.value=='(':
            left=self.expr();self.expect(')')
        elif t.kind=='NUMBER':left=('num',t.value)
        elif t.kind=='STRING':
            raw=t.value
            if raw.startswith("'"):
                raw='"'+raw[1:-1].replace('\\"','"').replace('"','\\"').replace("\\'", "'")+'"'
            try:left=('str',json.loads(raw))
            except ValueError:raise UUIError(f'Invalid string at {t.position}')
        elif t.value in ('true','false'):left=('bool',t.value=='true')
        elif t.kind=='NAME':left=('var',t.value)
        else:raise UUIError(f'Expected expression at {t.position}')
        precedence={'or':10,'and':20,'==':30,'!=':30,'<':30,'>':30,'<=':30,'>=':30,'+':40,'-':40,'*':50,'/':50}
        while True:
            if self.peek().value=='(' and 70>=min_prec:
                if left[0]!='var':raise UUIError('Only named functions can be called')
                self.take();args=[]
                if not self.accept(')'):
                    while True:
                        args.append(self.expr())
                        if self.accept(')'):break
                        self.expect(',')
                left=('call',left[1],args);continue
            op=self.peek().value
            if op not in precedence or precedence[op]<min_prec:break
            self.take();p=precedence[op];right=self.expr(p+1);left=('bin',op,left,right)
        return left

class ReturnSignal(Exception):
    def __init__(self,value):self.value=value

class Runtime:
    def __init__(self,bridge,max_statements=1000,max_tokens=20000,max_steps=10000,max_call_depth=32):
        self.bridge=bridge;self.vars={};self.output=[];self.functions={}
        self.max_statements=max_statements;self.max_tokens=max_tokens;self.max_steps=max_steps
        self.max_call_depth=max_call_depth;self.steps=0;self.depth=0
    def tick(self):
        self.steps+=1
        if self.steps>self.max_steps:raise UUIError('Execution step budget exceeded')
    @staticmethod
    def require_number(v):
        if type(v) is not Fraction:raise UUIError('Expected numeric value')
    @staticmethod
    def require_bool(v):
        if type(v) is not bool:raise UUIError('Expected boolean condition')
    @staticmethod
    def printable(v):return str(v).lower() if type(v) is bool else str(v)
    def evaluate(self,node,env=None):
        self.tick();env=self.vars if env is None else env
        kind=node[0]
        if kind=='num':return Fraction(node[1])
        if kind in ('str','bool'):return node[1]
        if kind=='var':
            if node[1] not in env:raise UUIError(f'Undefined variable {node[1]}')
            return env[node[1]]
        if kind=='unary':
            value=self.evaluate(node[2],env)
            if node[1]=='-':self.require_number(value);return -value
            self.require_bool(value);return not value
        if kind=='call':
            name=node[1]
            if name in COGNITIVE_OPS:
                args=[self.evaluate(a,env) for a in node[2]]
                try:return invoke_cognitive(name,args,self)
                except (ValueError,NotImplementedError,KeyError,TypeError) as e:raise UUIError(str(e)) from e
            if name not in self.functions:raise UUIError(f'Undefined function {name}')
            params,body=self.functions[name]
            if len(params)!=len(node[2]):raise UUIError(f'Wrong argument count for {name}')
            if self.depth>=self.max_call_depth:raise UUIError('Call depth budget exceeded')
            args=[self.evaluate(a,env) for a in node[2]]
            local=dict(zip(params,args));self.depth+=1
            try:
                try:self.execute(body,local,inside_fn=True)
                except ReturnSignal as result:return result.value
                return None
            finally:self.depth-=1
        if kind=='bin':
            op=node[1];a=self.evaluate(node[2],env)
            if op in ('and','or'):
                self.require_bool(a)
                if op=='and' and not a:return False
                if op=='or' and a:return True
                b=self.evaluate(node[3],env);self.require_bool(b);return b
            b=self.evaluate(node[3],env)
            if op in ('==','!='):
                result=type(a)==type(b) and a==b
                return result if op=='==' else not result
            self.require_number(a);self.require_number(b)
            try:
                return {'+':lambda:a+b,'-':lambda:a-b,'*':lambda:a*b,'/':lambda:a/b,
                        '<':lambda:a<b,'>':lambda:a>b,'<=':lambda:a<=b,'>=':lambda:a>=b}[op]()
            except ZeroDivisionError:raise UUIError('Division by zero')
        raise UUIError('Invalid expression')
    def execute(self,statements,env,inside_fn=False):
        for statement in statements:
            self.tick();op=statement[0]
            if op=='let':
                if statement[1] in env:raise UUIError(f'Variable already declared: {statement[1]}')
                env[statement[1]]=self.evaluate(statement[2],env)
            elif op=='set':
                if statement[1] not in env:raise UUIError(f'Undefined variable {statement[1]}')
                env[statement[1]]=self.evaluate(statement[2],env)
            elif op=='print':self.output.append(self.printable(self.evaluate(statement[1],env)))
            elif op=='observe':
                value=self.evaluate(statement[1],env);source=self.evaluate(statement[2],env)
                self.require_number(value)
                if not isinstance(source,str):raise UUIError('Observation source must be a string')
                self.output.append(self.bridge.numeric_observation(str(value),source=source))
            elif op=='resolve':
                a=self.evaluate(statement[1],env);b=self.evaluate(statement[2],env)
                self.require_number(a);self.require_number(b)
                if a.denominator!=1 or b.denominator!=1:raise UUIError('Forge IDs must be integers')
                try:self.output.append(self.bridge.forge_query(int(a),int(b)))
                except RuntimeError as e:raise UUIError(str(e)) from e
            elif op=='if':
                cond=self.evaluate(statement[1],env);self.require_bool(cond)
                self.execute(statement[2] if cond else statement[3],env,inside_fn)
            elif op=='while':
                while True:
                    self.tick();cond=self.evaluate(statement[1],env);self.require_bool(cond)
                    if not cond:break
                    self.execute(statement[2],env,inside_fn)
            elif op=='fn':
                if inside_fn:raise UUIError('Nested function definitions unsupported')
                name=statement[1]
                if name in self.functions or name in COGNITIVE_OPS:raise UUIError(f'Function already declared or reserved: {name}')
                self.functions[name]=(statement[2],statement[3])
            elif op=='return':
                if not inside_fn:raise UUIError('Return outside function')
                raise ReturnSignal(self.evaluate(statement[1],env))
    def run(self,source):
        tokens=tokenize(source)
        if len(tokens)>self.max_tokens:raise UUIError('Token budget exceeded')
        statements=Parser(source,tokens).program()
        if len(statements)>self.max_statements:raise UUIError('Statement budget exceeded')
        self.execute(statements,self.vars)
        return list(self.output)

    def run_compiled(self, statements):
        """Execute validated UUI IR. This is an AST interpreter, NOT a native bytecode VM."""
        if not isinstance(statements,list) or len(statements)>self.max_statements:
            raise UUIError('Invalid compiled program size')
        self.execute(statements,self.vars)
        return list(self.output)
