import json, os, statistics, tempfile, time
from numeric_macros import NumericMacroLearner as New
from numeric_macros_v03 import NumericMacroLearner as Old

def run(cls,n):
    fd,path=tempfile.mkstemp();os.close(fd)
    try:
        x=cls(path,max_period=8);v=0; x.observe('series',v)
        t=time.perf_counter()
        for i in range(n):
            v+=2 if i%2==0 else 3
            x.observe('series',v)
        ingest=time.perf_counter()-t
        t=time.perf_counter();rule=x.consolidate('series'); consolidate=time.perf_counter()-t
        t=time.perf_counter()
        for _ in range(100): result=x.predict('series')
        prediction=(time.perf_counter()-t)/100
        x.close(); size=os.stat(path).st_size
        return {'ingest_ms':round(ingest*1000,2),'consolidate_ms':round(consolidate*1000,2),'predict_ms':round(prediction*1000,4),'db_bytes':size,'period':rule['period'],'predicted':str(result)}
    finally: os.unlink(path)

if __name__=='__main__':
    result={}
    for n in (200,1000,3000):
        result[str(n)]={}
        for name,cls in [('v03',Old),('v04',New)]:
            trials=[run(cls,n) for _ in range(3)]
            result[str(n)][name]={k:(statistics.median(t[k] for t in trials) if isinstance(trials[0][k],float) else trials[0][k]) for k in trials[0]}
    print(json.dumps(result,indent=2))
    with open('benchmark_results.json','w') as f:json.dump(result,f,indent=2)
