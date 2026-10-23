import tempfile,unittest
from fractions import Fraction
from numeric_stream_v09 import NumericStreamV09
from numeric_stream_v10 import NumericStreamV10
class TestStreaming(unittest.TestCase):
 def test_equivalence(self):
  cases=[[0,2,5,7,10,12,15],list(range(70)),[0,1,3,8,9,17,21],list(range(100))+[1000], [Fraction(i,7) for i in range(60)]]
  for seq in cases:
   with tempfile.TemporaryDirectory() as t:
    a=NumericStreamV09(t+'/a');b=NumericStreamV10(t+'/b')
    a.observe_batch('x',seq);b.observe_batch('x',seq)
    self.assertEqual(a.predict('x'),b.predict('x'))
    self.assertEqual(a._rule('x'),b._rule('x'))
    a.close();b.close()
 def test_negative_cache_invalidation(self):
  c=NumericStreamV10();c.observe_batch('x',[0,1,3,6,10,15]);self.assertIsNone(c.predict('x'))
  self.assertIsNone(c.predict('x'));self.assertEqual(c.db.execute('SELECT period FROM v07_rules').fetchone()[0],0)
  c.observe('x',21);self.assertIsNone(c.predict('x'));c.close()
 def test_large_period(self):
  c=NumericStreamV10(max_period=16);v=0;seq=[v]
  for i in range(300):v+=(1,2,4,8,16)[i%5];seq.append(v)
  c.observe_batch('x',seq);self.assertEqual(c._rule('x')['period'],5);c.close()
if __name__=='__main__':unittest.main()
