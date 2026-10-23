import unittest, tempfile, os
from fractions import Fraction
from numeric_macros import NumericMacroLearner
class TestIncremental(unittest.TestCase):
 def test_long_period(self):
  x=NumericMacroLearner(max_period=4);v=Fraction(0)
  x.observe('a',v)
  for i in range(40):v+=Fraction([2,3][i%2]);x.observe('a',v)
  self.assertEqual(x.consolidate('a')['period'],2)
  self.assertEqual(x.predict('a'),v+2);x.close()
 def test_break_invalidate(self):
  x=NumericMacroLearner(max_period=2);v=0;x.observe('a',v)
  for d in [2,3]*8+[99]+[2,3]*8:v+=d;x.observe('a',v)
  self.assertIsNone(x.consolidate('a'));x.close()
 def test_reopen_and_upgrade(self):
  f=tempfile.NamedTemporaryFile(delete=False);f.close()
  try:
   x=NumericMacroLearner(f.name,max_period=3);v=0;x.observe('a',v)
   for d in [1,2]*8:v+=d;x.observe('a',v)
   x.consolidate('a');x.close()
   y=NumericMacroLearner(f.name,max_period=3);self.assertEqual(y.predict('a'),v+1)
   y.observe('a',v+1);self.assertEqual(y.consolidate('a')['period'],2);y.close()
  finally:os.unlink(f.name)
 def test_fractions(self):
  x=NumericMacroLearner();v=Fraction(0);x.observe('a',v)
  for i in range(16):v+=Fraction(1,3) if i%2==0 else Fraction(1,7);x.observe('a',v)
  x.consolidate('a');self.assertEqual(x.predict('a'),v+Fraction(1,3));x.close()
if __name__=='__main__':unittest.main()
