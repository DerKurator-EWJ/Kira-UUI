"""Reproducible workload comparison; measurements depend on host and SQLite config."""
import json, os, random, sqlite3, statistics, tempfile, time, tracemalloc
from fractions import Fraction
from pathlib import Path
from numeric_stream_v07 import NumericStream
from numeric_stream_v08 import NumericStreamV08
from numeric_stream_v09 import NumericStreamV09

MODES={'v09_cached_deferred':lambda p:NumericStreamV09(p),
       'v07_deferred':lambda p:NumericStream(p),
       'v08_deferred':lambda p:NumericStreamV08(p,mode='deferred'),
       'v08_incremental':lambda p:NumericStreamV08(p,mode='incremental')}

def values(kind,n,seed=42):
    rng=random.Random(seed); v=Fraction(0); out=[v]
    for i in range(1,n):
        if kind=='periodic':d=(2,3,5)[(i-1)%3]
        elif kind=='irregular':d=rng.randint(-30,30)
        elif kind=='regime':d=(2,3)[(i-1)%2] if i<n//2 else (7,11)[(i-n//2)%2]
        elif kind=='rational':d=Fraction((i-1)%3+1,7)
        else:raise ValueError(kind)
        v+=d;out.append(v)
    return out

def one(mode,kind,n,reads,series,batch,seed=42):
    with tempfile.TemporaryDirectory() as td:
        path=os.path.join(td,'data.sqlite'); core=MODES[mode](path)
        start=time.perf_counter_ns()
        for s in range(series):
            seq=values(kind,n,seed+s)
            name=f's{s}'
            if batch:core.observe_batch(name,seq)
            else:
                for v in seq:core.observe(name,v)
        learn=(time.perf_counter_ns()-start)/1e6
        start=time.perf_counter_ns()
        hits=0
        for j in range(reads):
            result=core.predict(f's{j%series}')
            hits+=result is not None
        query=(time.perf_counter_ns()-start)/1e6
        # Check answers on periodic/rational data only, and rejection for regime/irregular.
        expected=kind in ('periodic','rational')
        if bool(hits)!=expected or (expected and hits!=reads):
            raise AssertionError((mode,kind,n,reads,series,hits,expected))
        core.close()
        return {'learn_ms':learn,'read_ms':query,'total_ms':learn+query,'db_bytes':os.path.getsize(path),'prediction_hits':hits}

def run(repeats=3):
    cases=[
      ('periodic',1000,200,1,False),('periodic',1000,200,1,True),
      ('irregular',1000,200,1,True),('regime',1000,200,1,True),
      ('rational',1000,200,1,True),('periodic',100,200,50,True),
      ('irregular',100,200,50,True),('periodic',3000,10,1,True),
      ('periodic',3000,1000,1,True),
    ]
    output=[]
    for kind,n,reads,series,batch in cases:
        for mode in MODES:
            samples=[one(mode,kind,n,reads,series,batch) for _ in range(repeats)]
            output.append({'kind':kind,'observations_per_series':n,'reads':reads,'series':series,'batch':batch,'mode':mode,'repeats':repeats,**{k:round(statistics.median(x[k] for x in samples),3) for k in samples[0]}})
    return output

if __name__=='__main__':
    results=run()
    Path(__file__).with_name('benchmark_v09_results.json').write_text(json.dumps(results,indent=2))
    for row in results:print(row)
