"""Identical periodic workloads, same SQLite durability within each category.
Report median per-run TOTAL, not sum of phase medians. No batch vs single equivalence claim.
"""
import os,tempfile,time,statistics,json
from numeric_stream_v07 import NumericStream
from numeric_stream_v08 import NumericStreamV08
from numeric_cost_aware import CostAwareNumericLearner

def run(kind,n,reads,batch,interleaved=False):
 fd,path=tempfile.mkstemp();os.close(fd)
 if kind=='v06':x=CostAwareNumericLearner(path,max_period=4)
 elif kind=='v07':x=NumericStream(path,max_period=4)
 else:x=NumericStreamV08(path,max_period=4,mode=kind.split('_')[1])
 write=read=0.;last=None
 try:
  values=list(range(n))
  if interleaved:
   # equal number of events; prediction only after enough values for a valid macro
   blocks=[values[:n//2],values[n//2:]]
   for block in blocks:
    t=time.perf_counter()
    if batch and kind!='v06':x.observe_batch('s',block)
    else:
     for v in block:x.observe('s',v)
    write+=time.perf_counter()-t
    t=time.perf_counter()
    for _ in range(reads//2):last=x.predict('s')
    read+=time.perf_counter()-t
  else:
   t=time.perf_counter()
   if batch and kind!='v06':x.observe_batch('s',values)
   else:
    for v in values:x.observe('s',v)
   write=time.perf_counter()-t
   t=time.perf_counter()
   for _ in range(reads):last=x.predict('s')
   read=time.perf_counter()-t
  assert last==n, (kind,last,n)
  x.close()
  return {'write_ms':write*1000,'read_ms':read*1000,'total_ms':(write+read)*1000,'bytes':os.stat(path).st_size}
 finally:
  try:x.close()
  except Exception:pass
  os.unlink(path)

def bench(repeats=5):
 out={}
 for n,reads,interleaved in [(200,20,False),(1000,200,False),(1000,200,True)]:
  for batch in (False,True):
   for kind in (('v06','v07','v08_deferred','v08_incremental') if not batch else ('v07','v08_deferred','v08_incremental')):
    key=f'n{n}_r{reads}_interleaved{interleaved}_batch{batch}_{kind}'
    runs=[run(kind,n,reads,batch,interleaved) for _ in range(repeats)]
    out[key]={k:round(statistics.median(r[k] for r in runs),3) for k in runs[0]}
 return out
if __name__=='__main__':
 out=bench()
 with open('benchmark_v08_results.json','w') as f:json.dump(out,f,indent=2)
 print(json.dumps(out,indent=2))
