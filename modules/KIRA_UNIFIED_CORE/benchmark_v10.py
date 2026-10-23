"""Scaling tests. Each case isolated, timings exclude initialization and close. RSS is process peak, not per-core."""
import json,os,resource,statistics,tempfile,time,tracemalloc
from pathlib import Path
from numeric_stream_v09 import NumericStreamV09
from numeric_stream_v10 import NumericStreamV10

def sequence(n,kind):
 v=0
 yield v
 for i in range(n-1):
  d=(2,3,5)[i%3] if kind=='periodic' else ((i*1103515245+12345)%61)-30
  v+=d
  yield v

def run_one(cls,n,kind,reads=20):
 with tempfile.TemporaryDirectory() as td:
  c=cls(os.path.join(td,'d.sqlite'))
  t=time.perf_counter();c.observe_batch('s',sequence(n,kind));write=time.perf_counter()-t
  tracemalloc.start();t=time.perf_counter();pred=c.predict('s');first=time.perf_counter()-t
  _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
  t=time.perf_counter()
  for _ in range(reads-1):c.predict('s')
  repeat=time.perf_counter()-t
  size=os.path.getsize(os.path.join(td,'d.sqlite'));c.close()
  return dict(write_s=round(write,4),first_read_s=round(first,4),repeat_reads_s=round(repeat,4),consolidation_peak_python_bytes=peak,db_bytes=size,prediction=str(pred))

def run():
 out=[]
 for n in (10000,100000,1000000):
  for kind in ('periodic','irregular'):
   for label,cls in [('v09',NumericStreamV09),('v10',NumericStreamV10)]:
    row=dict(n=n,kind=kind,mode=label,**run_one(cls,n,kind));out.append(row);print(row,flush=True)
 Path(__file__).with_name('benchmark_v10_results.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':run()
