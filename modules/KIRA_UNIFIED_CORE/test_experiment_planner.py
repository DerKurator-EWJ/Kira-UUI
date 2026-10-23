import sqlite3
import unittest
from evidence_layer import EvidenceLayer
from inquiry_controller import InquiryController
from experiment_planner import ExperimentPlanner

class ExperimentPlannerTests(unittest.TestCase):
    def setUp(self):
        self.db=sqlite3.connect(':memory:')
        self.ev=EvidenceLayer(self.db)
        self.inquiry=InquiryController(self.ev)
        self.planner=ExperimentPlanner(self.inquiry)
        self.q=self.inquiry.ask_about('bed','edible')
        self.planner.register_capability('inspect','edible','Inspect the material',risk=0.1)
        self.planner.register_capability('touch','edible','Try a safe simulated interaction',risk=0.3)
    def tearDown(self): self.db.close()
    def test_rank_and_dedupe(self):
        first=self.planner.propose(self.q)
        self.assertEqual([x['capability_id'] for x in first],['inspect','touch'])
        self.assertEqual([x['experiment_id'] for x in first],
                         [x['experiment_id'] for x in self.planner.propose(self.q)])
    def test_approval_and_record(self):
        x=self.planner.propose(self.q)[0]['experiment_id']
        with self.assertRaises(ValueError): self.planner.complete(x,event_id='e1',value=False,source_ref='sim')
        self.planner.approve(x)
        r=self.planner.complete(x,event_id='e1',value=False,source_ref='sim')
        self.assertFalse(r['needs_review'])
        self.assertEqual(self.ev.evidence_for('bed','edible')[0]['value'],False)
        with self.assertRaises(ValueError): self.planner.complete(x,event_id='e2',value=True,source_ref='sim')
    def test_contradiction_creates_inquiry(self):
        self.ev.record(event_id='before',subject='bed',predicate='edible',value=False,source_ref='sim')
        x=self.planner.propose(self.q)[0]['experiment_id']; self.planner.approve(x)
        r=self.planner.complete(x,event_id='after',value=True,source_ref='sim')
        self.assertTrue(r['needs_review'])
        self.assertTrue(any(q['kind']=='contradiction' for q in self.inquiry.next_questions()))
    def test_no_matching_capability(self):
        other=self.inquiry.ask_about('chair','weight')
        self.assertEqual(self.planner.propose(other),[])
    def test_disabled_and_invalid_risk(self):
        self.planner.register_capability('off','edible','Disabled',enabled=False)
        self.assertEqual(len(self.planner.propose(self.q)),2)
        with self.assertRaises(ValueError): self.planner.register_capability('bad','edible','Bad',risk=float('nan'))
    def test_cancel(self):
        x=self.planner.propose(self.q)[0]['experiment_id']
        self.planner.cancel(x)
        with self.assertRaises(ValueError): self.planner.approve(x)
