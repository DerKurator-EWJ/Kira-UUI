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
 def test_experiment_requires_approval(self):
  b=UUIBridge()
  q=b.ask_about('bed','edible')
  b.register_capability('inspect','edible','Simulated inspection',risk=0.1)
  proposals=b.propose_experiments(q)
  self.assertEqual(len(proposals),1)
  eid=proposals[0]['experiment_id']
  with self.assertRaises(ValueError):b.record_verified_experiment(eid,event_id='sim1',value=False,source_ref='sim')
  b.approve_experiment(eid)
  b.record_verified_experiment(eid,event_id='sim1',value=False,source_ref='sim')
  self.assertEqual(b.evidence.evidence_for('bed','edible')[0]['value'],False)
  b.close()
if __name__=='__main__':unittest.main()
