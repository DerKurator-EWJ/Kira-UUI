import tempfile, os, time, statistics, json
from numeric_adaptive import AdaptiveNumericLearner
from numeric_cost_aware import CostAwareNumericLearner

def bench(cls,n,reads):
    values=[]
    for _ in range(3):
        fd,path=tempfile.mkstemp();os.close(fd)
        x=cls(path,max_period=4)
        try:
            t=time.perf_counter()
            for i in range(n):x.observe('s',i)
            learn=(time.perf_counter()-t)*1000
            t=time.perf_counter()
            for _ in range(reads):x.predict('s')
            query=(time.perf_counter()-t)*1000
            values.append((learn,query,x.mode('s')))
        finally:x.close();os.unlink(path)
    return {'learn_ms_median':round(statistics.median(a for a,b,c in values),3), 'predict_ms_median':round(statistics.median(b for a,b,c in values),3),'final_mode':values[-1][2]}
if __name__=='__main__':
    out={}
    for n,r in [(100,100),(1000,200)]:
        out[f'{n}_observations_{r}_reads']={'v05':bench(AdaptiveNumericLearner,n,r),'v06':bench(CostAwareNumericLearner,n,r)}
    with open('benchmark_v06_results.json','w') as f:json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))
