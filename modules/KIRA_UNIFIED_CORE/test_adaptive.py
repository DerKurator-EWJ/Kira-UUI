import unittest, tempfile, os
from fractions import Fraction
from numeric_adaptive import AdaptiveNumericLearner
class AdaptiveTests(unittest.TestCase):
 def test_promote_persist_and_extend(self):
  f=tempfile.NamedTemporaryFile(delete=False);f.close()
  try:
   a=AdaptiveNumericLearner(f.name,promote_reads=3,max_period=4)
   v=Fraction(0);a.observe('s',v)
   for i in range(20):v+=Fraction([2,3][i%2]);a.observe('s',v)
   self.assertEqual(a.mode('s'),'deferred')
   self.assertEqual(a.predict('s'),v+2)
   self.assertEqual(a.predict('s'),v+2)
   self.assertEqual(a.predict('s'),v+2)
   self.assertEqual(a.mode('s'),'incremental')
   a.observe('s',v+2);self.assertEqual(a.predict('s'),v+5);a.close()
   b=AdaptiveNumericLearner(f.name,promote_reads=3,max_period=4)
   self.assertEqual(b.mode('s'),'incremental')
   self.assertEqual(b.predict('s'),v+5);b.close()
  finally:os.unlink(f.name)
 def test_independent_streams_and_empty(self):
  a=AdaptiveNumericLearner(promote_reads=2,max_period=2)
  self.assertIsNone(a.predict('missing'))
  a.observe('x',0);a.observe('y',10)
  for i in range(1,14):a.observe('x',i);a.observe('y',10+i*2)
  a.predict('x');a.predict('x')
  self.assertEqual(a.mode('x'),'incremental');self.assertEqual(a.mode('y'),'deferred')
  self.assertEqual(a.predict('y'),38);a.close()
if __name__=='__main__':unittest.main()
