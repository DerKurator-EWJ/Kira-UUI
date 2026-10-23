import os,tempfile,unittest
from decision_matrix import DecisionMatrix,EvaluationInput
from evaluation_bridge import EvaluatedNumericStream

class MatrixTests(unittest.TestCase):
    def sample(self,surprise,valence,success,scope=0):
        return EvaluationInput(.95,surprise,success,.8,.8,.5,valence,contradiction_scope=scope)
    def test_reactions(self):
        m=DecisionMatrix()
        for args,expected in [((.1,.7,.9),'GEIL'),((.9,.8,.9),'NOTGEIL'),((.9,-.8,0),'FUCK'),((.95,0,.5,1),'CANTOR')]:
            self.assertEqual(m.evaluate(self.sample(*args)).primary,expected)
    def test_validation(self):
        with self.assertRaises(ValueError):self.sample(1.5,0,0)
    def test_stream_persistence_and_prediction(self):
        with tempfile.TemporaryDirectory() as td:
            path=os.path.join(td,'core.db')
            c=EvaluatedNumericStream(path)
            def feed(i,value):return c.observe_and_evaluate(str(i),'sequence',value,success=.9,novelty=.1,relevance=.7,information_gain=.2,valence=.7)
            for i in range(10):feed(i,i)
            self.assertEqual(c.core.predict('sequence'),10)
            r=feed(10,100)
            self.assertEqual(r['primary'],'NOTGEIL')
            self.assertTrue(r['evidence_review_required'])
            with self.assertRaises(ValueError):feed(10,100)
            c.close()
            c=EvaluatedNumericStream(path)
            self.assertEqual(len(c.history('sequence')),11)
            self.assertEqual(c.history('sequence')[-1]['predicted'],'10/1')
            c.close()
    def test_no_prediction_no_false_surprise(self):
        c=EvaluatedNumericStream()
        r=c.observe_and_evaluate('a','x',1,success=.5,novelty=.8,relevance=.5,information_gain=.1,valence=0)
        self.assertEqual(r['surprise'],0)
        self.assertEqual(r['expectation_confidence'],0)
        c.close()
if __name__=='__main__':unittest.main()
