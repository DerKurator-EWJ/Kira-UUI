"""RRM runtime 0.1: shared binary arena + evidence/continuation.
Experimental bridge, NOT a native binary execution engine.
"""
import json
import sqlite3
from rrm_id import Record, encode, decode, KIND_NODE, KIND_RELATION, KIND_STATE, INLINE, COMPOSITE
from rrm05 import validate, changes, neighborhood

class BinaryRuntime:
    def __init__(self, path=':memory:'):
        self.db=sqlite3.connect(path)
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS records(id INTEGER PRIMARY KEY, wire BLOB NOT NULL UNIQUE);
        CREATE TABLE IF NOT EXISTS observations(stream TEXT, position INTEGER, state_id INTEGER NOT NULL REFERENCES records(id), PRIMARY KEY(stream,position));
        CREATE TABLE IF NOT EXISTS evidence(stream TEXT, signature TEXT, context TEXT, outcome TEXT, count INTEGER NOT NULL, PRIMARY KEY(stream,signature,context,outcome));
        ''')
        self.db.commit()
        self.cache={}

    def _put(self, rec):
        wire=encode(rec)
        self.db.execute('INSERT OR IGNORE INTO records(wire) VALUES (?)',(wire,))
        return self.db.execute('SELECT id FROM records WHERE wire=?',(wire,)).fetchone()[0]

    def _get(self, ref):
        row=self.db.execute('SELECT wire FROM records WHERE id=?',(ref,)).fetchone()
        if row is None: raise ValueError('dangling reference')
        rec,end=decode(row[0]); assert end==len(row[0])
        return rec

    def _atom(self, text):
        raw=str(text).encode('utf-8')
        return self._put(Record(KIND_NODE,INLINE,int.from_bytes(raw,'little'),len(raw)*8))

    def _string(self, ref):
        rec=self._get(ref)
        if rec.kind!=KIND_NODE or rec.mode!=INLINE or rec.bits%8: raise ValueError('invalid atom')
        return rec.value.to_bytes(rec.bits//8,'little').decode('utf-8')

    def store_state(self,state):
        validate(state)
        # Canonical node ordering and stable references; repeated labels interned.
        if True:  # caller controls transaction; necessary for speculative savepoints
            nodes=sorted(state['nodes'],key=str)
            node_refs={name:self._put(Record(KIND_RELATION,COMPOSITE,refs=(self._atom(name),self._atom(state['nodes'][name])))) for name in nodes}
            edge_refs=[self._put(Record(KIND_RELATION,COMPOSITE,refs=(node_refs[a],self._atom(rel),node_refs[b]))) for a,rel,b in state['edges']]
            # Explicit node count preserves isolated nodes. Edges retain input order (aligned transitions).
            return self._put(Record(KIND_STATE,COMPOSITE,refs=(len(nodes),*map(node_refs.get,nodes),*edge_refs)))

    def load_state(self,ref):
        rec=self._get(ref)
        if rec.kind!=KIND_STATE or rec.mode!=COMPOSITE: raise ValueError('not a state')
        count=rec.refs[0]; nodes={}; by_ref={}
        for node_ref in rec.refs[1:count+1]:
            pair=self._get(node_ref)
            if pair.kind!=KIND_RELATION or len(pair.refs)!=2: raise ValueError('invalid node')
            name,kind=map(self._string,pair.refs)
            nodes[name]=kind; by_ref[node_ref]=name
        edges=[]
        for edge_ref in rec.refs[count+1:]:
            edge=self._get(edge_ref)
            if edge.kind!=KIND_RELATION or len(edge.refs)!=3: raise ValueError('invalid edge')
            a,r,b=edge.refs
            if a not in by_ref or b not in by_ref: raise ValueError('invalid endpoint')
            edges.append([by_ref[a],self._string(r),by_ref[b]])
        result={'nodes':nodes,'edges':edges}
        validate(result)
        return result

    def observe(self,stream,state):
        validate(state)
        with self.db:
            row=self.db.execute('SELECT position,state_id FROM observations WHERE stream=? ORDER BY position DESC LIMIT 1',(stream,)).fetchone()
            position=row[0]+1 if row else 0
            if row:
                before=self.load_state(row[1])
                for sig,ctx,outcome in changes(before,state):
                    self.db.execute('''INSERT INTO evidence VALUES (?,?,?,?,1)
                        ON CONFLICT(stream,signature,context,outcome) DO UPDATE SET count=count+1''',
                        (stream,json.dumps(sig),json.dumps(ctx),outcome))
            ref=self.store_state(state)
            self.db.execute('INSERT INTO observations VALUES (?,?,?)',(stream,position,ref))
        return position

    def predict(self,stream,state,focus=None,min_support=2,contextual=True):
        # Same evidence semantics as RRM-KERN 0.5; input can be a state handle.
        if isinstance(state,int): state=self.load_state(state)
        validate(state)
        candidates=[]
        for i,(a,rel,b) in enumerate(state['edges']):
            if focus is not None and focus!=i: continue
            sig=json.dumps((str(state['nodes'][a]),str(rel),str(state['nodes'][b])))
            ctx=json.dumps(neighborhood(state,i))
            if contextual:
                rows=self.db.execute('SELECT outcome,count FROM evidence WHERE stream=? AND signature=? AND context=?',(stream,sig,ctx)).fetchall()
            else:
                rows=self.db.execute('SELECT outcome,SUM(count) FROM evidence WHERE stream=? AND signature=? GROUP BY outcome',(stream,sig)).fetchall()
            if not rows: continue
            rows.sort(key=lambda r:(-r[1],r[0])); total=sum(n for _,n in rows)
            if total<min_support: continue
            if len(rows)>1 and rows[0][1]==rows[1][1]:
                return {'status':'OPEN','reason':'tied outcomes','alternatives':rows}
            candidates.append((i,rows,total))
        if len(candidates)!=1:
            return {'status':'OPEN','reason':'no unique supported target','candidates':len(candidates)}
        i,rows,total=candidates[0]; outcome,support=rows[0]
        result=json.loads(json.dumps(state))
        if outcome!='NO_CHANGE': result['edges'][i][1]=outcome
        # Pure prediction: don't persist generated state or evidence.
        return {'status':'HYPOTHESIS','result':result,'support':support,'total':total,
                'frequency':support/total,'contextual':contextual,'alternatives':rows}

    def dream(self,stream,state,steps=3,**kwargs):
        if isinstance(state,int): state=self.load_state(state)
        out=[]
        for _ in range(steps):
            p=self.predict(stream,state,**kwargs); out.append(p)
            if p['status']!='HYPOTHESIS': break
            state=p['result']
        return out

    def stats(self):
        return {name:self.db.execute('SELECT COUNT(*) FROM '+name).fetchone()[0] for name in ('records','observations','evidence')}

    def close(self): self.db.close()
