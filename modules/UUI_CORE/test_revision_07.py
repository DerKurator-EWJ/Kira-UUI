import json, tempfile, unittest
from pathlib import Path
from uui_bridge import UUIBridge
from uui_lang import Runtime, UUIError

class RevisionTests(unittest.TestCase):
    def test_propose_and_apply_persist(self):
        with tempfile.TemporaryDirectory() as d:
            db=str(Path(d)/'mem.sqlite')
            b=UUIBridge(kira_db=db)
            try:
                vm=Runtime(b)
                vm.run('hier lern_draus("A","farbe","rot","person","e1")\n'
                       'hier lern_draus("A","farbe","blau","camera","e2","observation",3/4)')
                rid=b.evidence.pending_reviews()[0]['review_id']
                p=json.loads(vm.run(f'hier pruef_revision("{rid}")')[-1])
                self.assertFalse(p['automatic_resolution'])
                self.assertEqual(p['suggested'],'retain_both')
                self.assertEqual(len(b.evidence.evidence_for('A','farbe')),2)
                with self.assertRaises(UUIError):vm.run(f'hier entscheide_revision("{rid}","reject_new","no")')
                vm.run(f'hier entscheide_revision("{rid}","retain_both","insufficient independent evidence")')
                self.assertEqual(len(b.evidence.pending_reviews()),0)
                self.assertEqual(len(b.evidence.evidence_for('A','farbe')),2)
            finally:b.close()
            b=UUIBridge(kira_db=db)
            try:
                vm=Runtime(b)
                history=json.loads(vm.run(f'hier revision_verlauf("{rid}")')[-1])
                self.assertEqual(len(history),1)
                self.assertEqual(history[0]['decision'],'retain_both')
            finally:b.close()
    def test_supersede_does_not_delete(self):
        b=UUIBridge()
        try:
            vm=Runtime(b)
            vm.run('hier lern_draus("B","status","alt","a","b1")\n'
                   'hier lern_draus("B","status","neu","b","b2")')
            rid=b.evidence.pending_reviews()[0]['review_id']
            vm.run(f'hier entscheide_revision("{rid}","supersede_old","verified by independent measurement")')
            rows=b.evidence.evidence_for('B','status')
            self.assertEqual([x['status'] for x in rows],['superseded','active'])
        finally:b.close()
if __name__=='__main__':unittest.main()
