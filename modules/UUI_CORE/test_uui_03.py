import unittest
from pathlib import Path
from uui_bridge import UUIBridge
from uui_lang import Runtime,UUIError
from uui_compiler import compile_source,serialize,deserialize

class WilliTests(unittest.TestCase):
    def setUp(self):self.bridge=UUIBridge()
    def tearDown(self):self.bridge.close()
    def test_source_and_compiled_equal(self):
        src=Path(__file__).with_name('willi_demo.uui').read_text(encoding='utf-8')
        a=Runtime(self.bridge).run(src)
        b=Runtime(self.bridge).run_compiled(deserialize(serialize(compile_source(src))))
        self.assertEqual(a[0],'Geil!');self.assertEqual(a[2:],['5','4']);self.assertEqual(a[0],b[0]);self.assertEqual(a[1]['rational'],b[1]['rational']);self.assertEqual(a[2:],b[2:])
    def test_else(self):
        self.assertEqual(Runtime(self.bridge).run('was_brauchst_du 1 > 2 {\n hier "x"\n} geht_das {\n hier "y"\n}'),['y'])
    def test_integrity(self):
        blob=serialize(compile_source('hier 1'))
        with self.assertRaisesRegex(UUIError,'integrity'):
            deserialize(blob.replace(b'"1"',b'"2"'))
    def test_invalid_ast(self):
        import json,hashlib
        program=[['os_system','echo bad']]
        body=json.dumps(program,ensure_ascii=False,separators=(',',':'))
        obj={'format':'uui.uuix.ast.v1','sha256':hashlib.sha256(body.encode()).hexdigest(),'program':program}
        with self.assertRaisesRegex(UUIError,'Unsupported'):deserialize(json.dumps(obj).encode())
if __name__=='__main__':unittest.main()
