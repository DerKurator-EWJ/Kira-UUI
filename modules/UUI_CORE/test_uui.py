import unittest
from uui_bridge import UUIBridge
from uui_lang import Runtime,UUIError

class UUITests(unittest.TestCase):
    def setUp(self):self.bridge=UUIBridge();self.vm=Runtime(self.bridge)
    def tearDown(self):self.bridge.close()
    def test_exact_arithmetic(self):
        self.assertEqual(self.vm.run('let x = 1 / 3\nprint x + 2 / 3'),['1'])
    def test_precedence(self):self.assertEqual(self.vm.run('print 1 + 2 * 3'),['7'])
    def test_parentheses(self):self.assertEqual(self.vm.run('print (1 + 2) * 3'),['9'])
    def test_observation(self):
        rows=self.vm.run('observe 1/3 as "camera"')
        self.assertEqual(rows[0]['rational'],['1','3'])
        self.assertEqual(self.bridge.numeric.evidence(1)[0][0],'camera')
    def test_missing_forge_pack(self):
        with self.assertRaisesRegex(UUIError,'forge pack not loaded'):self.vm.run('resolve 2, 3')
    def test_undefined_variable(self):
        with self.assertRaisesRegex(UUIError,'Undefined variable'):self.vm.run('print x')
    def test_zero_division(self):
        with self.assertRaisesRegex(UUIError,'Division by zero'):self.vm.run('print 1 / 0')
    def test_reject_python(self):
        with self.assertRaises(UUIError):self.vm.run('__import__("os")')
    def test_budget(self):
        with self.assertRaisesRegex(UUIError,'Statement budget'):Runtime(self.bridge,max_statements=1).run('print 1\nprint 2')
    def test_string(self):self.assertEqual(self.vm.run('print "Hallo UUI"'),['Hallo UUI'])
    def test_line_separator(self):self.assertEqual(self.vm.run('let x = 5; print x'),['5'])
    def test_noninteger_id(self):
        with self.assertRaisesRegex(UUIError,'must be integers'):self.vm.run('resolve 1/2, 3')
if __name__=='__main__':unittest.main()
