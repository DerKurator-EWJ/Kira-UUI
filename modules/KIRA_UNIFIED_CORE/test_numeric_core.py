import tempfile, os, unittest
from fractions import Fraction
from numeric_core import NumericCore, rational

class NumericTests(unittest.TestCase):
    def test_exact_range(self):
        c=NumericCore()
        ids=[c.observe(s) for s in ('0.125','0.126','0.128','0.731')]
        self.assertEqual([i for i,_ in c.range('0.127','0.001')],ids[1:3]);c.close()
    def test_id_not_distance(self):
        c=NumericCore();a=c.observe('1000');b=c.observe('0.1')
        self.assertEqual(b,a+1)
        self.assertEqual([i for i,_ in c.range('0.1','0')],[b]);c.close()
    def test_persistent_expansion(self):
        with tempfile.TemporaryDirectory() as d:
            p=os.path.join(d,'data.sqlite');c=NumericCore(p)
            i=c.observe('1/3',source='camera');c.observe('2/6',identity=i,source='text');c.close()
            c=NumericCore(p)
            self.assertEqual(c.value(i),Fraction(1,3));self.assertEqual(len(c.evidence(i)),2)
            self.assertEqual(c.range('1/3',0),[(i,Fraction(1,3))]);c.close()
    def test_conflict_not_overwritten(self):
        c=NumericCore();i=c.observe('1/2');c.observe('2/3',identity=i)
        self.assertEqual(len(c.evidence(i)),2)
        self.assertEqual(c.value(i),Fraction(1,2));c.close()
    def test_reject_float(self):
        with self.assertRaises(TypeError): rational(.1)
    def test_negative_tolerance(self):
        c=NumericCore()
        with self.assertRaises(ValueError):c.range(1,-1)
        c.close()
if __name__=='__main__':unittest.main()
