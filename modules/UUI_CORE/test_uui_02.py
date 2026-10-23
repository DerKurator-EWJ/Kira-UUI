import unittest
from pathlib import Path
from uui_bridge import UUIBridge
from uui_lang import Runtime,UUIError
class ControlFlowTests(unittest.TestCase):
    def setUp(self):self.bridge=UUIBridge();self.vm=Runtime(self.bridge)
    def tearDown(self):self.bridge.close()
    def test_function_loop_branch(self):
        src=Path(__file__).with_name('control_flow.uui').read_text(encoding='utf-8')
        result=self.vm.run(src)
        self.assertEqual(result[0],'UUI 0.2 OK')
        self.assertEqual(result[1]['rational'],['30','1'])
    def test_if_followed_by_statement(self):
        self.assertEqual(self.vm.run('if true {\n print 1\n}\nprint 2'),['1','2'])
    def test_boolean_logic(self):
        self.assertEqual(self.vm.run('print true and not false\nprint 3 < 4 or false'),['true','true'])
    def test_loop_budget(self):
        with self.assertRaisesRegex(UUIError,'step budget'):
            Runtime(self.bridge,max_steps=30).run('while true {\n print 1\n}')
    def test_recursive_limit(self):
        with self.assertRaisesRegex(UUIError,'Call depth'):
            Runtime(self.bridge,max_call_depth=4).run('fn f(x) {\n return f(x)\n}\nprint f(1)')
    def test_function_scope(self):
        with self.assertRaisesRegex(UUIError,'Undefined variable'):
            self.vm.run('fn f(x) {\n return x\n}\nprint f(1)\nprint x')
    def test_type_check(self):
        with self.assertRaisesRegex(UUIError,'boolean'):
            self.vm.run('if 1 {\n print 1\n}')
    def test_undefined_set(self):
        with self.assertRaisesRegex(UUIError,'Undefined variable'):
            self.vm.run('set x = 1')
    def test_no_host_execution(self):
        with self.assertRaises(UUIError):self.vm.run('print __import__("os")')
if __name__=='__main__':unittest.main()
