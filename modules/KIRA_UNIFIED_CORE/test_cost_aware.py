import unittest, tempfile, os
from fractions import Fraction
from numeric_cost_aware import CostAwareNumericLearner

class CostAwareTests(unittest.TestCase):
 def test_persistence_and_exactness(self):
  f=tempfile.NamedTemporaryFile(delete=False);f.close()
  try:
   x=CostAwareNumericLearner(f.name,predicted_future_reads=0)
   v=Fraction(1,3);x.observe('s',v)
   for i in range(40):v+=Fraction([1,2][i%2],3);x.observe('s',v)
   self.assertEqual(x.predict('s'),v+Fraction(1,3))
   self.assertEqual(x.mode('s'),'deferred');x.close()
   x=CostAwareNumericLearner(f.name,predicted_future_reads=0)
   self.assertEqual(x.predict('s'),v+Fraction(1,3));x.close()
  finally:os.unlink(f.name)
 def test_no_macro_when_irregular(self):
  x=CostAwareNumericLearner(predicted_future_reads=0)
  for v in [0,1,4,10,11,29,34,35,49]:x.observe('x',v)
  self.assertIsNone(x.predict('x'));x.close()
 def test_independent_series(self):
  x=CostAwareNumericLearner(predicted_future_reads=0)
  for i in range(30):x.observe('a',i);x.observe('b',2*i)
  self.assertEqual(x.predict('a'),30)
  self.assertEqual(x.predict('b'),60)
  x.close()
 def test_promote_with_cost_evidence(self):
  x=CostAwareNumericLearner(max_period=2,predicted_future_reads=10000,
    cost_scan_per_delta=1000,cost_incremental_per_delta=.01,cost_replay_per_delta=.01)
  for i in range(30):x.observe('a',i)
  self.assertEqual(x.predict('a'),30)
  self.assertEqual(x.mode('a'),'incremental')
  x.observe('a',30)
  self.assertEqual(x.predict('a'),31)
  x.close()
if __name__=='__main__':unittest.main()
