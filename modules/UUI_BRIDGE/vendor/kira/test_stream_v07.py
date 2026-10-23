import unittest,tempfile,os
from fractions import Fraction
from numeric_stream_v07 import NumericStream
class TestStream(unittest.TestCase):
 def test_one_shot_persistent_extend(self):
  f=tempfile.NamedTemporaryFile(delete=False);f.close()
  try:
   x=NumericStream(f.name);x.observe_batch('a',[0,2,5,7,10,12,15]);self.assertEqual(x.predict('a'),17);x.close()
   x=NumericStream(f.name);self.assertEqual(x.predict('a'),17);x.observe('a',17);self.assertEqual(x.predict('a'),20);x.close()
  finally:os.unlink(f.name)
 def test_batch_vs_single(self):
  a=NumericStream();b=NumericStream()
  seq=[Fraction(i,3) for i in range(100)]
  for v in seq:a.observe('x',v)
  b.observe_batch('x',seq)
  self.assertEqual(a.predict('x'),b.predict('x'));a.close();b.close()
 def test_irregular(self):
  x=NumericStream();x.observe_batch('x',[0,1,4,10,11,29,34,35,49]);self.assertIsNone(x.predict('x'));x.close()
 def test_multiple_series(self):
  x=NumericStream();x.observe_batch('a',range(30));x.observe_batch('b',(i*2 for i in range(30)))
  self.assertEqual(x.predict('a'),30);self.assertEqual(x.predict('b'),60);x.close()
 def test_invalidates_rule(self):
  x=NumericStream();x.observe_batch('a',range(30));self.assertEqual(x.predict('a'),30)
  x.observe('a',999);self.assertIsNone(x.predict('a'));x.close()
if __name__=='__main__':unittest.main()
