"""Direct operations over RRM-ID record references (SQLite-backed arena).
No JSON state materialization in these operations. Immutable snapshots.
Aligned edges and node identities are explicit preconditions for DIFF.
"""
from dataclasses import dataclass
from rrm_id import Record, KIND_RELATION, KIND_STATE, COMPOSITE

@dataclass(frozen=True)
class Delta:
    index: int
    old_edge: int
    new_edge: int

class BinaryOps:
    def __init__(self, runtime):
        self.rt=runtime

    def _state(self, handle):
        rec=self.rt._get(handle)
        if rec.kind!=KIND_STATE or rec.mode!=COMPOSITE or not rec.refs:
            raise ValueError('invalid state')
        n=rec.refs[0]
        if n<0 or n>len(rec.refs)-1: raise ValueError('invalid node count')
        return rec, n

    def relate(self, source, relation_type, target):
        """Intern relation directly using existing record handles."""
        for ref in (source,relation_type,target): self.rt._get(ref)
        return self.rt._put(Record(KIND_RELATION,COMPOSITE,refs=(source,relation_type,target)))

    def diff(self, before, after):
        """Detect aligned edge replacements by comparing encoded handles.
        Returns None if nodes/edge layout incompatible, including reordered edges.
        """
        a,n=self._state(before); b,m=self._state(after)
        if n!=m or len(a.refs)!=len(b.refs) or a.refs[1:n+1]!=b.refs[1:m+1]:
            return None
        result=[]
        for i,(old,new) in enumerate(zip(a.refs[n+1:],b.refs[m+1:])):
            if old==new: continue
            o=self.rt._get(old); p=self.rt._get(new)
            if o.kind!=KIND_RELATION or p.kind!=KIND_RELATION or len(o.refs)!=3 or len(p.refs)!=3:
                return None
            if o.refs[0]!=p.refs[0] or o.refs[2]!=p.refs[2]: return None
            result.append(Delta(i,old,new))
        return tuple(result)

    def continue_with(self, state, delta):
        """Apply a previously observed replacement of relation type to an aligned edge.
        Reuses immutable nodes and unaffected edges; does not persist an observation.
        """
        rec,n=self._state(state)
        index=delta.index
        edges=list(rec.refs[n+1:])
        if index<0 or index>=len(edges): raise IndexError(index)
        current=self.rt._get(edges[index]); old=self.rt._get(delta.old_edge); new=self.rt._get(delta.new_edge)
        if any(r.kind!=KIND_RELATION or len(r.refs)!=3 for r in (current,old,new)):
            raise ValueError('invalid relation')
        # Strict guard: same relation type as observed, same endpoint node types;
        # new endpoint names are allowed if node kinds match.
        if current.refs[1]!=old.refs[1]: raise ValueError('inapplicable relation')
        for j in (0,2):
            current_node=self.rt._get(current.refs[j]); old_node=self.rt._get(old.refs[j])
            if len(current_node.refs)!=2 or len(old_node.refs)!=2 or current_node.refs[1]!=old_node.refs[1]:
                raise ValueError('inapplicable endpoint type')
        edges[index]=self.relate(current.refs[0],new.refs[1],current.refs[2])
        return self.rt._put(Record(KIND_STATE,COMPOSITE,refs=rec.refs[:n+1]+tuple(edges)))
