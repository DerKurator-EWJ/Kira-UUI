import json
import unittest
from uui_bridge import UUIBridge
from uui_lang import Runtime, UUIError

class Cognition05Tests(unittest.TestCase):
    def setUp(self):
        self.bridge=UUIBridge()
        self.runtime=Runtime(self.bridge)
    def tearDown(self):self.bridge.close()
    def run_uui(self,src):return self.runtime.run(src)
    def test_evidence_and_provenance(self):
        out=self.run_uui('hier lern_draus("A","farbe","rot","sensor","event1")\n'
                         'hier was_weisst_du("A","farbe")\n'
                         'hier woher_weisst_du("A","farbe")')
        self.assertTrue(json.loads(out[0])['recorded'])
        self.assertEqual(json.loads(out[1])['evidence'][0]['value'],'rot')
        self.assertEqual(json.loads(out[2])['sources'][0]['source_ref'],'sensor')
    def test_conflict_retained_and_inquiry(self):
        out=self.run_uui('hier lern_draus("A","farbe","rot","sensor","e1")\n'
                         'hier lern_draus("A","farbe","blau","sensor","e2")\n'
                         'hier was_weisst_du("A","farbe")')
        self.assertTrue(json.loads(out[1])['needs_review'])
        self.assertEqual(len(json.loads(out[2])['evidence']),2)
        self.assertTrue(any(q['kind']=='contradiction' for q in self.bridge.inquiry.next_questions()))
    def test_unknown_inquiry_deduplicated(self):
        out=self.run_uui('hier warum_nicht("A","gewicht")\n'
                         'hier warum_nicht("A","gewicht")')
        self.assertEqual(json.loads(out[0])['inquiry_id'],json.loads(out[1])['inquiry_id'])
    def test_counterfactual_no_mutation(self):
        self.assertEqual(self.run_uui('hier was_waere_wenn(1 / 3, 2 / 3, "plus")'),['1'])
        self.assertEqual(self.bridge.evidence.evidence_for('A','farbe'),[])
    def test_no_silent_inference(self):
        self.assertFalse(json.loads(self.run_uui('hier was_weisst_du("A","farbe")')[0])['known'])
    def test_duplicate_event_rejected(self):
        with self.assertRaises(UUIError):
            self.run_uui('hier lern_draus("A","farbe","rot","sensor","e1")\n'
                         'hier lern_draus("A","farbe","rot","sensor","e1")')

if __name__=='__main__':unittest.main()
