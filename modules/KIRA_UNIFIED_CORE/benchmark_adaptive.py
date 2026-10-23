import time, tempfile, os, json
from numeric_adaptive import AdaptiveNumericLearner
from numeric_macros import NumericMacroLearner as Fast
from numeric_macros_v03 import NumericMacroLearner as Slow

def run(cls,reads=0,n=1000):
 fd,path=tempfile.mkstemp(suffix='.db');os.close(fd)
 try:
  x=cls(path,max_period=4) if cls!=AdaptiveNumericLearner else cls(path,max_period=4,promote_reads=8)
  t=time.perf_counter();v=0;x.observe('s',v)
  for i in range(n):v+=2 if i%2==0 else 3;x.observe('s',v)
  learn=time.perf_counter()-t
  x.consolidate('s');t=time.perf_counter()
  for _ in range(reads):x.predict('s')
  predict=time.perf_counter()-t
  mode=x.mode('s') if cls==AdaptiveNumericLearner else cls.__name__
  x.close()
  return {'learn_ms':round(learn*1000,3),'predict_total_ms':round(predict*1000,3),'mode':mode}
 finally:os.unlink(path)
if __name__=='__main__':
 results={str(r):{c.__name__+('_'+c.__module__):run(c,r) for c in (Slow,Fast,AdaptiveNumericLearner)} for r in (1,20,200)}
 print(json.dumps(results,indent=2))
 with open('benchmark_adaptive_results.json','w') as f:json.dump(results,f,indent=2)
