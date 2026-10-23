import unittest
from uui_lang import Runtime,UUIError
from uui_bridge import UUIBridge
from uui_cognition import FAMILIES,ALL
class CognitiveTests(unittest.TestCase):
 def setUp(self):self.bridge=UUIBridge();self.rt=Runtime(self.bridge)
 def tearDown(self):self.bridge.close()
 def test_registry(self):self.assertEqual(len(FAMILIES),13);self.assertEqual(len(ALL),39)
 def test_questions(self):self.assertEqual(self.rt.run('hier warum_nicht("A")'),['Welche Alternative oder Gegenbegruendung gibt es zu: A'])
 def test_exact_identity(self):self.assertEqual(self.rt.run('hier ist_das_noch(1 / 3, 2 / 6)'),['true'])
 def test_confidence(self):self.assertEqual(self.rt.run('hier wie_sicher(1 / 2)'),['true'])
 def test_observe(self):self.assertIn('kira.numeric:1',self.rt.run('hier merk_dir(1 / 3, "test")')[0])
 def test_recall(self):self.assertEqual(self.rt.run('es_sei x = merk_dir(3 / 5, "test")\nhier erinner_dich(1)')[-1],'3/5')
 def test_unimplemented_explicit(self):
  with self.assertRaisesRegex(UUIError,'not implemented'):self.rt.run('hier guck()')
 def test_permission_denied(self):self.assertEqual(self.rt.run('hier wer_darf("camera")'),['false'])
 def test_stop(self):
  with self.assertRaisesRegex(UUIError,'stopped'):self.rt.run('hier stopp()')
 def test_cannot_shadow_builtin(self):
  with self.assertRaises(UUIError):self.rt.run('mach warum(x) { her_damit x }')
 def test_compiled_equivalence(self):
  from uui_compiler import compile_source,serialize,deserialize
  src='hier warum("A")'
  self.assertEqual(self.rt.run_compiled(deserialize(serialize(compile_source(src)))),['Welche Begruendung fehlt fuer: A'])
if __name__=='__main__':unittest.main()
