import unittest
from fractions import Fraction
from uui_bridge import ScopedIdentity,UUIBridge
class BridgeTests(unittest.TestCase):
 def test_scoped_ids(self):
  self.assertNotEqual(ScopedIdentity('forge.subject',1).key(),ScopedIdentity('kira.numeric',1).key())
 def test_exact_numeric(self):
  b=UUIBridge(); r=b.numeric_observation('1/3');self.assertEqual(r['rational'],['1','3']);b.close()
 def test_conflict_retained(self):
  b=UUIBridge();a=b.observe_claim(event_id='a',subject='bed',predicate='edible',value=False)
  c=b.observe_claim(event_id='b',subject='bed',predicate='edible',value=True)
  self.assertTrue(c['needs_review']);self.assertEqual(len(b.evidence.pending_reviews()),1);b.close()
 def test_no_pack(self):
  b=UUIBridge()
  with self.assertRaises(RuntimeError):b.forge_query(1,2)
  b.close()
if __name__=='__main__':unittest.main()
