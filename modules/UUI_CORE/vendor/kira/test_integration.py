import tempfile, os, unittest
from efficient_core import EfficientCore
class Integration(unittest.TestCase):
 def test_numeric_and_token_share_persistence(self):
  with tempfile.TemporaryDirectory() as d:
   p=os.path.join(d,'kira.db');k=EfficientCore(p)
   i=k.observe_numeric('0.25',source='sensor');k.observe('s',42);k.close()
   k=EfficientCore(p)
   self.assertEqual(k.query_numeric('0.25',0)[0][0],i)
   self.assertEqual(k.numeric.evidence(i)[0][0],'sensor');k.close()
if __name__=='__main__':unittest.main()
