import os,tempfile,unittest
from fractions import Fraction
from benchmark_v09 import values,MODES,one

class WorkloadTests(unittest.TestCase):
    def test_series_shapes(self):
        for kind in ('periodic','irregular','regime','rational'):
            self.assertEqual(len(values(kind,100)),100)
    def test_exact_rational_prediction(self):
        seq=values('rational',100)
        expected=seq[-1]+Fraction(1,7) # index 99 -> next step 1/7
        for factory in MODES.values():
            c=factory(':memory:');c.observe_batch('r',seq)
            self.assertEqual(c.predict('r'),expected);c.close()
    def test_no_fabricated_periodicity(self):
        for kind in ('irregular','regime'):
            for factory in MODES.values():
                c=factory(':memory:');c.observe_batch('r',values(kind,1000))
                self.assertIsNone(c.predict('r'));c.close()
    def test_persistence_multiseries(self):
        for factory in MODES.values():
            with tempfile.TemporaryDirectory() as td:
                p=os.path.join(td,'x.sqlite');c=factory(p)
                for j in range(10):c.observe_batch(str(j),values('periodic',100,seed=j))
                predictions=[c.predict(str(j)) for j in range(10)];c.close()
                c=factory(p)
                self.assertEqual([c.predict(str(j)) for j in range(10)],predictions);c.close()
    def test_negative_cache_invalidates(self):
        from numeric_stream_v09 import NumericStreamV09
        c=NumericStreamV09(':memory:')
        c.observe_batch('x',[0,1,3,6,10,15,21])
        self.assertIsNone(c.predict('x'))
        self.assertEqual(c.db.execute('SELECT period FROM v07_rules WHERE name=?',('x',)).fetchone()[0],0)
        c.observe_batch('x',[28,36])
        self.assertIsNone(c.predict('x'))
        self.assertEqual(c.db.execute('SELECT version FROM v07_rules WHERE name=?',('x',)).fetchone(),c.db.execute('SELECT version FROM v07_series WHERE name=?',('x',)).fetchone())
        c.close()
    def test_benchmark_oracle(self):
        for mode in MODES:
            for kind in ('periodic','irregular','regime','rational'):
                one(mode,kind,60,10,2,True)

if __name__=='__main__':unittest.main()
