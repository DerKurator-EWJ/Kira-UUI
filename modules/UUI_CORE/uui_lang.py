"""UUI Core 0.1: small, deliberately restricted interpreted language.

No Python eval/exec, host imports, loops, file I/O or native calls.
"""
from dataclasses import dataclass
from fractions import Fraction
import re

class UUIError(Exception):
    pass

@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    position: int

TOKEN = re.compile(r'''(?P<SPACE>[ \t\r]+)|(?P<COMMENT>\#[^\n]*)|(?P<NEWLINE>\n)|(?P<NUMBER>\d+(?:\.\d+)?)|(?P<STRING>"(?:\\["\\nrt]|[^"\\])*"|'(?:\\['\\nrt]|[^'\\])*')|(?P<NAME>[A-Za-z_][A-Za-z_0-9]*)|(?P<SYM>[=()+*/;,\-])|(?P<INVALID>.)''')

def tokenize(source):
    result=[]
    for m in TOKEN.finditer(source):
        k=m.lastgroup
        if k in ('SPACE','COMMENT'): continue
        if k=='INVALID': raise UUIError(f'Unexpected character {m.group()!r} at {m.start()}')
        result.append(Token(k,m.group(),m.start()))
    result.append(Token('EOF','',len(source)))
    return result

class Parser:
    def __init__(self,source): self.tokens=tokenize(source);self.i=0
    def peek(self): return self.tokens[self.i]
    def take(self): t=self.peek();self.i+=1;return t
    def accept(self,value):
        if self.peek().value==value:self.take();return True
        return False
    def expect(self,value):
        if not self.accept(value):raise UUIError(f'Expected {value!r} at {self.peek().position}')
    def name(self):
        t=self.take()
        if t.kind!='NAME':raise UUIError(f'Expected identifier at {t.position}')
        return t.value
    def separators(self):
        n=0
        while self.peek().kind=='NEWLINE' or self.accept(';'):
            if self.peek().kind=='NEWLINE':self.take()
            n+=1
        return n
    def program(self):
        out=[];self.separators()
        while self.peek().kind!='EOF':
            t=self.take()
            if t.kind!='NAME' or t.value not in ('let','print','observe','resolve'):
                raise UUIError(f'Unknown statement {t.value!r} at {t.position}')
            if t.value=='let':
                name=self.name();self.expect('=');stmt=('let',name,self.expr())
            elif t.value=='print':stmt=('print',self.expr())
            elif t.value=='observe':
                value=self.expr()
                self.expect('as');source=self.expr();stmt=('observe',value,source)
            else:
                subject=self.expr();self.expect(',');relation=self.expr();stmt=('resolve',subject,relation)
            out.append(stmt)
            if self.peek().kind!='EOF' and not self.separators():
                raise UUIError(f'Expected newline or semicolon at {self.peek().position}')
        return out
    def expr(self,min_prec=0):
        t=self.take()
        if t.value=='-':left=('neg',self.expr(30))
        elif t.value=='(':
            left=self.expr();self.expect(')')
        elif t.kind=='NUMBER':left=('num',t.value)
        elif t.kind=='STRING':
            import json
            raw=t.value
            if raw.startswith("'"):
                # Single-quoted strings use the same limited escapes.
                raw='"'+raw[1:-1].replace('\\"','"').replace('"','\\"').replace("\\'", "'")+'"'
            try:left=('str',json.loads(raw))
            except ValueError:raise UUIError(f'Invalid string at {t.position}')
        elif t.kind=='NAME':left=('var',t.value)
        else:raise UUIError(f'Expected expression at {t.position}')
        precs={'+':10,'-':10,'*':20,'/':20}
        while self.peek().value in precs and precs[self.peek().value]>=min_prec:
            op=self.take().value;p=precs[op]
            right=self.expr(p+1);left=('bin',op,left,right)
        return left

class Runtime:
    def __init__(self,bridge,max_statements=1000,max_tokens=20000):
        self.bridge=bridge;self.vars={};self.output=[]
        self.max_statements=max_statements;self.max_tokens=max_tokens
    def evaluate(self,node):
        kind=node[0]
        if kind=='num':return Fraction(node[1])
        if kind=='str':return node[1]
        if kind=='var':
            if node[1] not in self.vars:raise UUIError(f'Undefined variable {node[1]}')
            return self.vars[node[1]]
        if kind=='neg':
            v=self.evaluate(node[1]);self.require_number(v);return -v
        if kind=='bin':
            a=self.evaluate(node[2]);b=self.evaluate(node[3]);self.require_number(a);self.require_number(b)
            op=node[1]
            try:return {'+':lambda:a+b,'-':lambda:a-b,'*':lambda:a*b,'/':lambda:a/b}[op]()
            except ZeroDivisionError:raise UUIError('Division by zero')
        raise UUIError('Invalid expression')
    @staticmethod
    def require_number(v):
        if not isinstance(v,Fraction):raise UUIError('Expected numeric value')
    @staticmethod
    def printable(v):
        if isinstance(v,Fraction):return str(v)
        return str(v)
    def run(self,source):
        tokens=tokenize(source)
        if len(tokens)>self.max_tokens:raise UUIError('Token budget exceeded')
        stmts=Parser(source).program()
        if len(stmts)>self.max_statements:raise UUIError('Statement budget exceeded')
        for statement in stmts:
            op=statement[0]
            if op=='let':
                _,name,expr=statement
                if name in ('let','print','observe','resolve','as'):raise UUIError('Reserved identifier')
                self.vars[name]=self.evaluate(expr)
            elif op=='print':self.output.append(self.printable(self.evaluate(statement[1])))
            elif op=='observe':
                value=self.evaluate(statement[1]);source=self.evaluate(statement[2]);self.require_number(value)
                if not isinstance(source,str):raise UUIError('Observation source must be a string')
                self.output.append(self.bridge.numeric_observation(str(value),source=source))
            elif op=='resolve':
                a=self.evaluate(statement[1]);b=self.evaluate(statement[2]);self.require_number(a);self.require_number(b)
                if a.denominator!=1 or b.denominator!=1:raise UUIError('Forge IDs must be integers')
                try:self.output.append(self.bridge.forge_query(int(a),int(b)))
                except RuntimeError as e:raise UUIError(str(e)) from e
        return list(self.output)
