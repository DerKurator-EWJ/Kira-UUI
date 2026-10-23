import unittest
from evaluation_bridge import EvaluatedNumericStream

class EvidenceTests(unittest.TestCase):
    def setUp(self):self.k=EvaluatedNumericStream()
    def tearDown(self):self.k.close()
    def add(self,event,value,context=None):
        return self.k.observe_and_evaluate(event,'observations',len(event),success=.5,novelty=.5,
          relevance=.5,information_gain=.5,valence=0,fact=dict(subject='bed',predicate='edible',
          value=value,context=context or {'room':'A'},source_type='action_result',source_ref='avatar',confidence=.9))
    def test_conflict_does_not_overwrite(self):
        self.add('first',False)
        r=self.add('second',True)
        self.assertTrue(r['evidence_review_required'])
        self.assertEqual(len(self.k.evidence.pending_reviews()),1)
        self.assertEqual([e['status'] for e in self.k.evidence.evidence_for('bed','edible')],['active','active'])
        rid=self.k.evidence.pending_reviews()[0]['review_id']
        self.k.evidence.resolve(rid,'supersede_old')
        self.assertEqual([e['status'] for e in self.k.evidence.evidence_for('bed','edible')],['superseded','active'])
    def test_context_matters(self):
        self.add('first',False,{'mode':'normal'})
        self.add('second',True,{'mode':'modified'})
        self.assertEqual(self.k.evidence.pending_reviews(),[])
    def test_source_validation(self):
        with self.assertRaises(ValueError):self.k.evidence.record(event_id='x',subject='x',predicate='x',value=True,source_type='invented',source_ref='z')
    def test_no_fact_still_works(self):
        r=self.k.observe_and_evaluate('a','s',1,success=.5,novelty=.5,relevance=.5,information_gain=.5,valence=0)
        self.assertIsNone(r['evidence'])
if __name__=='__main__':unittest.main()
