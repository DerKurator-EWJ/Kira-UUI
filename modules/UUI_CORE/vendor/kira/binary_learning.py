"""Experimental binary evidence engine. Shared immutable arena; no JSON evidence keys.
Precondition: aligned edge order and node labels across consecutive observations.
"""
from rrm_id import Record, KIND_TRANSITION, KIND_RELATION, COMPOSITE
from unified import UnifiedRuntime

class BinaryLearningRuntime(UnifiedRuntime):
    def __init__(self, path=':memory:', cache_budget=65536):
        super().__init__(path, cache_budget)
        self.db.execute('''CREATE TABLE IF NOT EXISTS binary_evidence (
          stream TEXT NOT NULL, signature INTEGER NOT NULL REFERENCES records(id),
          context INTEGER NOT NULL REFERENCES records(id), outcome INTEGER NOT NULL REFERENCES records(id),
          count INTEGER NOT NULL CHECK(count>0),
          PRIMARY KEY(stream,signature,context,outcome))''')
        self.db.commit()

    def _edge_parts(self, handle):
        edge=self._get(handle)
        if edge.kind != KIND_RELATION or edge.mode != COMPOSITE or len(edge.refs)!=3:
            raise ValueError('invalid edge')
        a,r,b=edge.refs
        na,nb=self._get(a),self._get(b)
        if len(na.refs)!=2 or len(nb.refs)!=2:
            raise ValueError('invalid node')
        return a,r,b,na.refs[1],nb.refs[1]

    def _features(self, state, index):
        rec,n=self.ops._state(state)
        edges=rec.refs[n+1:]
        a,r,b,ak,bk=self._edge_parts(edges[index])
        signature=self._put(Record(KIND_TRANSITION,COMPOSITE,refs=(ak,r,bk)))
        # Neighborhood is a sorted multiset of relation-role signatures.
        # Include adjacent edges only, and encode their relative endpoints without names.
        parts=[]
        for j,ref in enumerate(edges):
            if j==index: continue
            x,t,y,xk,yk=self._edge_parts(ref)
            if x not in (a,b) and y not in (a,b):continue
            role_x=1 if x==a else 2 if x==b else 3
            role_y=1 if y==a else 2 if y==b else 3
            item=self._put(Record(KIND_TRANSITION,COMPOSITE,refs=(role_x,xk,t,role_y,yk)))
            parts.append(item)
        context=self._put(Record(KIND_TRANSITION,COMPOSITE,refs=tuple(sorted(parts))))
        return signature,context

    def _record_evidence(self, stream, previous, current):
        diff=self.ops.diff(previous,current)
        if diff is None or len(diff)>1:return
        old,n=self.ops._state(previous)
        old_edges=old.refs[n+1:]
        indices=[diff[0].index] if diff else range(len(old_edges))
        for i in indices:
            signature,context=self._features(previous,i)
            if diff:
                outcome=self._edge_parts(diff[0].new_edge)[1]
            else:
                # Distinct sentinel from every real relation type.
                outcome=self._put(Record(KIND_TRANSITION,COMPOSITE,refs=()))
            self.db.execute('''INSERT INTO binary_evidence VALUES (?,?,?,?,1)
                ON CONFLICT(stream,signature,context,outcome) DO UPDATE SET count=count+1''',
                (stream,signature,context,outcome))

    def observe(self,stream,state):
        # Atomic state storage, evidence and observation append.
        with self.db:
            previous=self.db.execute('SELECT position,state_id FROM observations WHERE stream=? ORDER BY position DESC LIMIT 1',(stream,)).fetchone()
            handle=self.store_state(state) if not isinstance(state,int) else state
            self.ops._state(handle)
            position=previous[0]+1 if previous else 0
            if previous:self._record_evidence(stream,previous[1],handle)
            self.db.execute('INSERT INTO observations VALUES (?,?,?)',(stream,position,handle))
        return position

    def predict(self,stream,state,focus=None,min_support=2,contextual=True):
        handle=self.store_state(state) if not isinstance(state,int) else state
        rec,n=self.ops._state(handle)
        edges=rec.refs[n+1:]
        candidates=[]
        for i,edge in enumerate(edges):
            if focus is not None and i!=focus:continue
            sig,ctx=self._features(handle,i)
            if contextual:
                rows=self.db.execute('SELECT outcome,count FROM binary_evidence WHERE stream=? AND signature=? AND context=?',(stream,sig,ctx)).fetchall()
            else:
                rows=self.db.execute('SELECT outcome,SUM(count) FROM binary_evidence WHERE stream=? AND signature=? GROUP BY outcome',(stream,sig)).fetchall()
            if not rows:continue
            rows.sort(key=lambda x:(-x[1],x[0]));total=sum(c for _,c in rows)
            if total<min_support:continue
            if len(rows)>1 and rows[0][1]==rows[1][1]:
                return {'status':'OPEN','reason':'tied outcomes','alternatives':rows}
            candidates.append((i,rows,total))
        if len(candidates)!=1:
            return {'status':'OPEN','reason':'no unique supported target','candidates':len(candidates)}
        i,rows,total=candidates[0]
        outcome,support=rows[0]
        sentinel=self._get(outcome)
        if sentinel.kind==KIND_TRANSITION and sentinel.mode==COMPOSITE and sentinel.refs==():
            result_handle=handle
        else:
            a,old,b,_,_=self._edge_parts(edges[i])
            newedge=self.ops.relate(a,outcome,b)
            newedges=list(edges);newedges[i]=newedge
            result_handle=self._put(Record(rec.kind,COMPOSITE,refs=rec.refs[:n+1]+tuple(newedges)))
        return {'status':'HYPOTHESIS','result':self.load_state(result_handle),
                'result_handle':result_handle,'support':support,'total':total,
                'frequency':support/total,'contextual':contextual,'alternatives':rows}

    def dream(self,stream,state,steps=3,**kwargs):
        # Speculative records exist only within the savepoint, then are rolled back.
        # No observation/evidence writes and no persistent speculative records.
        self.db.execute('SAVEPOINT rrm_dream')
        try:
            results=[]
            for _ in range(steps):
                p=self.predict(stream,state,**kwargs)
                results.append(p)
                if p['status']!='HYPOTHESIS':break
                state=p['result_handle']
            # Handles in the returned predictions are ephemeral; remove them.
            for p in results:p.pop('result_handle',None)
            return results
        finally:
            self.db.execute('ROLLBACK TO SAVEPOINT rrm_dream')
            self.db.execute('RELEASE SAVEPOINT rrm_dream')
            self._wires.clear();self._cache_bytes=0
