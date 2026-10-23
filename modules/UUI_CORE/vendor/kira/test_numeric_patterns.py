import tempfile, unittest
from fractions import Fraction
from pathlib import Path
from numeric_core import NumericCore,BlockIndex
from numeric_patterns import NumericPatternLearner

class NumericLearningTests(unittest.TestCase):
    def test_block_index_sorted_and_duplicates(self):
        b=BlockIndex(4)
        for i in list(range(30,0,-1))+[5,5]: b.insert((Fraction(i),i))
        self.assertEqual(len(list(b.between((Fraction(5),-1),(Fraction(5),float('inf'))))),3)
        self.assertEqual([x[0] for x in b.between((Fraction(3),-1),(Fraction(7),float('inf')))],list(map(Fraction,[3,4,5,5,5,6,7])))
    def test_numeric_range_many_insertions(self):
        c=NumericCore()
        for i in range(1000,0,-1): c.observe(i)
        self.assertEqual([v for _,v in c.range(500,2)],list(map(Fraction,[498,499,500,501,502])))
        c.close()
    def test_delta_reuse_across_series(self):
        p=NumericPatternLearner()
        for v in [0,2,5,7,10]: p.observe('training',v)
        p.observe('unseen',100);p.observe('unseen',102)
        self.assertEqual(p.predict('unseen')['value'],Fraction(105))
        p.close()
    def test_persistent_incremental_and_uncertainty(self):
        with tempfile.TemporaryDirectory() as d:
            path=str(Path(d)/'learning.sqlite')
            p=NumericPatternLearner(path)
            for v in [0,2,5,7]: p.observe('a',v)
            p.close();p=NumericPatternLearner(path)
            self.assertEqual(p.predict('a')['value'],Fraction(10))
            p.observe('a',10); self.assertGreaterEqual(p.predict('a')['support'],1)
            p.close()
    def test_unknown_not_fabricated(self):
        p=NumericPatternLearner();p.observe('x',5);p.observe('x',7)
        self.assertIsNone(p.predict('x'))
        p.close()
    def test_fraction_exactness(self):
        p=NumericPatternLearner()
        for v in ['0','1/3','2/3','1']: p.observe('x',v)
        self.assertEqual(p.predict('x')['value'],Fraction(4,3))
        p.close()

if __name__=='__main__': unittest.main()
