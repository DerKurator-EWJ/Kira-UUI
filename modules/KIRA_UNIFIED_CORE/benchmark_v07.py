import tempfile,os,time,json,statistics
from numeric_stream_v07 import NumericStream
from numeric_macros_v03 import NumericMacroLearner as Old

def run(cls,n=1000,reads=200,batch=False):
 f=tempfile.NamedTemporaryFile(delete=False);f.close()
 try:
  x=cls(f.name);seq=[i for i in range(n)]
  t=time.perf_counter()
  if batch:x.observe_batch('s',seq)
  else:
   for v in seq:x.observe('s',v)
  learn=time.perf_counter()-t
  t=time.perf_counter()
  for i in range(reads):
   if cls is not NumericStream and i==0:x.consolidate('s')
   p=x.predict('s')
   assert p==n
  pred=time.perf_counter()-t
  x.close();return {'learn_ms':learn*1000,'read_ms':pred*1000,'total_ms':(learn+pred)*1000,'db_bytes':os.path.getsize(f.name)}
 finally:os.unlink(f.name)
if __name__=='__main__':
 results={k:[run(cls,batch=batch) for _ in range(5)] for k,cls,batch in [('v03',Old,False),('v07_single',NumericStream,False),('v07_batch',NumericStream,True)]}
 print(json.dumps({k:{m:round(statistics.median(r[m] for r in rows),3) for m in rows[0]} for k,rows in results.items()},indent=2))
