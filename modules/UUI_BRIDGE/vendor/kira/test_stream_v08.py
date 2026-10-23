import unittest,tempfile,os
from fractions import Fraction
from numeric_stream_v08 import NumericStreamV08
class TestStream08(unittest.TestCase):
 def test_modes_and_batch(self):
  for mode in ('deferred','incremental'):
   x=NumericStreamV08(mode=mode,max_period=4)
   self.assertEqual(x.observe_batch('s',[0,2,5,7,10,12,15]),7)
   self.assertEqual(x.predict('s'),17)
   x.observe('s',17)
   self.assertEqual(x.predict('s'),20)
   x.close()
 def test_reject_exception(self):
  for mode in ('deferred','incremental'):
   x=NumericStreamV08(mode=mode,max_period=3)
   x.observe_batch('s',[0,1,2,3,4,5,7,8,9,10,11])
   self.assertIsNone(x.predict('s'))
   x.close()
 def test_restart_and_rationals(self):
  f=tempfile.NamedTemporaryFile(delete=False);f.close()
  try:
   for mode in ('incremental','incremental'):
    x=NumericStreamV08(f.name,mode=mode)
    if x.db.execute('SELECT COUNT(*) FROM v07_series').fetchone()[0]==0:
     x.observe_batch('s',[Fraction(i,3) for i in range(12)])
    self.assertEqual(x.predict('s'),Fraction(12,3))
    x.close()
  finally:os.unlink(f.name)
 def test_independent_series(self):
  x=NumericStreamV08(mode='incremental')
  x.observe_batch('a',list(range(10)))
  x.observe_batch('b',[3*i for i in range(10)])
  self.assertEqual(x.predict('a'),10)
  self.assertEqual(x.predict('b'),30)
  x.close()
 def test_no_history_read_after_consolidation(self):
  x=NumericStreamV08(mode='incremental')
  x.observe_batch('s',list(range(30)))
  self.assertEqual(x.predict('s'),30)
  x.db.execute('DELETE FROM v07_observations WHERE name=?',('s',))
  self.assertEqual(x.predict('s'),30)
  x.close()
if __name__=='__main__':unittest.main()
