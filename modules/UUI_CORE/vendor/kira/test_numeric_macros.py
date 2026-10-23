import os, tempfile, unittest
from fractions import Fraction
from numeric_macros import NumericMacroLearner

class TestNumericMacros(unittest.TestCase):
    def test_periodic_persist_predict(self):
        with tempfile.TemporaryDirectory() as d:
            p=os.path.join(d,'test.sqlite')
            a=NumericMacroLearner(p)
            v=Fraction(0)
            a.observe('a',v)
            for i in range(20):
                v += (Fraction(2,3),Fraction(5,7))[i%2]
                a.observe('a',v)
            a.consolidate('a')
            self.assertEqual(a.rule('a')['period'],2)
            self.assertEqual(a.predict('a'),v+Fraction(2,3))
            a.close()
            b=NumericMacroLearner(p)
            self.assertEqual(b.predict('a'),v+Fraction(2,3))
            b.close()

    def test_no_macro_when_not_profitable(self):
        a=NumericMacroLearner()
        for x in [0,1,2,3,4]:a.observe('s',x)
        a.consolidate('s')
        self.assertIsNone(a.rule('s'))
        a.close()

    def test_no_false_macro_on_irregular(self):
        a=NumericMacroLearner()
        for x in [0,1,3,6,10,15,21,28,36,45]:a.observe('s',x)
        a.consolidate('s')
        self.assertIsNone(a.rule('s'))
        a.close()

    def test_no_cross_series_contamination(self):
        a=NumericMacroLearner()
        for s, step in [('a',1),('b',3)]:
            for i in range(15):a.observe(s,i*step)
        a.consolidate('a'); a.consolidate('b')
        self.assertEqual(a.predict('a'),15)
        self.assertEqual(a.predict('b'),45)
        a.close()

    def test_exact_fraction(self):
        a=NumericMacroLearner()
        for i in range(20):a.observe('s',Fraction(i,11))
        a.consolidate('s')
        self.assertEqual(a.predict('s'),Fraction(20,11))
        a.close()

if __name__=='__main__':unittest.main()
