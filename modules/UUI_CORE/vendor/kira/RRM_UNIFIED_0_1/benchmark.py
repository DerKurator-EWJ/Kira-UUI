"""Repeatable end-to-end comparison; includes SQLite and Python overhead."""
import tempfile, time, json, os
from unified import UnifiedRuntime
from runtime import BinaryRuntime

def state(i):
    return {'nodes':{'a':'point','b':'point'},'edges':[['a','near' if i%2==0 else 'far','b']]}

def run(cls, n=600, q=1200, **kwargs):
    with tempfile.TemporaryDirectory() as d:
        rt=cls(os.path.join(d,'test.sqlite'),**kwargs)
        start=time.perf_counter()
        for i in range(n): rt.observe('x',state(i))
        write=time.perf_counter()-start
        handles=[x[0] for x in rt.db.execute('SELECT state_id FROM observations ORDER BY position')]
        start=time.perf_counter()
        for i in range(q): rt._get(handles[i%len(handles)])
        read=time.perf_counter()-start
        start=time.perf_counter()
        for i in range(100): rt.predict('x',state(i),contextual=False)
        predict=time.perf_counter()-start
        out={'observe_s':round(write,5),'read_s':round(read,5),'predict_s':round(predict,5),
             'db_bytes':os.path.getsize(os.path.join(d,'test.sqlite'))}
        if hasattr(rt,'cache_stats'): out['cache']=rt.cache_stats()
        rt.close()
        return out

if __name__=='__main__':
    print(json.dumps({'baseline':run(BinaryRuntime),'unified_0':run(UnifiedRuntime,cache_budget=0),
        'unified_64KiB':run(UnifiedRuntime,cache_budget=65536)},indent=2))
