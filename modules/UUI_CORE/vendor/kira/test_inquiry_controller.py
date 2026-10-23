import sqlite3
import unittest
from evidence_layer import EvidenceLayer
from inquiry_controller import InquiryController
from evaluation_bridge import EvaluatedNumericStream

class InquiryTests(unittest.TestCase):
    def setUp(self):
        self.db=sqlite3.connect(':memory:')
        self.e=EvidenceLayer(self.db)
        self.q=InquiryController(self.e)
    def tearDown(self):self.db.close()
    def test_curiosity_without_surprise(self):
        ident=self.q.ask_about('Stuhl','Verwendung')
        self.assertEqual(self.q.ask_about('Stuhl','Verwendung'),ident)
        self.assertEqual(len(self.q.next_questions()),1)
        self.assertEqual(self.q.next_questions()[0]['kind'],'unknown')
    def test_conflict_and_provenance(self):
        for i,v in enumerate((False,True)):
            self.e.record(event_id=f'ev{i}',subject='Bett',predicate='essbar',value=v,
                          context={'room':'home'},source_ref='sensor')
        self.q.sync_reviews()
        self.assertEqual(len(self.q.next_questions()),1)
        item=self.q.next_questions()[0]
        self.assertEqual(item['kind'],'contradiction')
        self.assertEqual(len(item['evidence_refs']),2)
        self.q.answer(item['inquiry_id'],'experiment:followup')
        self.assertFalse(self.q.next_questions())
        self.assertEqual(len(self.e.pending_reviews()),1)  # no silent resolution
    def test_persistence(self):
        ident=self.q.ask_about('Keks','essbar')
        self.assertEqual(InquiryController(self.e).next_questions()[0]['inquiry_id'],ident)
    def test_priority(self):
        self.q.ask_about('Bett','Material',priority=.1)
        self.q.ask_about('Keks','Form',priority=.8)
        self.assertEqual(self.q.next_questions(1)[0]['subject'],'Keks')
    def test_bridge(self):
        s=EvaluatedNumericStream()
        try:
            self.assertIsInstance(s.inquiry,InquiryController)
        finally:s.close()

if __name__=='__main__':unittest.main()
