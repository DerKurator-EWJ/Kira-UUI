import json
import tempfile
import unittest
from pathlib import Path
from uui_bridge import UUIBridge
from uui_lang import Runtime, UUIError

class PersistenceAndProvenance06(unittest.TestCase):
    def test_restart_and_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            path=str(Path(folder)/'kira.sqlite')
            bridge=UUIBridge(kira_db=path)
            try:
                vm=Runtime(bridge)
                out=vm.run('hier lern_draus("A","farbe","rot","kamera1","event1","observation",3 / 4)')
                self.assertEqual(json.loads(out[0])['source_type'],'observation')
                self.assertEqual(json.loads(out[0])['confidence'],0.75)
            finally: bridge.close()
            bridge=UUIBridge(kira_db=path)
            try:
                vm=Runtime(bridge)
                records=json.loads(vm.run('hier was_weisst_du("A","farbe")')[0])['evidence']
                self.assertEqual(len(records),1)
                self.assertEqual(records[0]['source_type'],'observation')
                self.assertEqual(records[0]['confidence'],0.75)
                self.assertEqual(records[0]['source_ref'],'kamera1')
                self.assertEqual(json.loads(vm.run('hier such_widerspruch()')[-1])['reviews'],[])
            finally: bridge.close()
    def test_conflicts_survive_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            path=str(Path(folder)/'kira.sqlite')
            bridge=UUIBridge(kira_db=path)
            try:
                vm=Runtime(bridge)
                vm.run('hier lern_draus("A","farbe","rot","user","e1","external_claim")\n'
                       'hier lern_draus("A","farbe","blau","kamera","e2","observation")')
            finally: bridge.close()
            bridge=UUIBridge(kira_db=path)
            try:
                vm=Runtime(bridge)
                reviews=json.loads(vm.run('hier such_widerspruch()')[-1])['reviews']
                self.assertEqual(len(reviews),1)
                audit=json.loads(vm.run('hier pruef_dich("A","farbe")')[-1])
                self.assertEqual(audit['verdict'],'requires_review')
                self.assertEqual(audit['evidence_count'],2)
            finally:bridge.close()
    def test_reject_invalid_provenance_and_confidence(self):
        bridge=UUIBridge()
        try:
            vm=Runtime(bridge)
            with self.assertRaises(UUIError):
                vm.run('hier lern_draus("A","p","v","src","e","unknown")')
            with self.assertRaises(UUIError):
                vm.run('hier lern_draus("A","p","v","src","e","observation",3 / 2)')
            self.assertEqual(bridge.evidence.evidence_for('A','p'),[])
        finally:bridge.close()

if __name__=='__main__':unittest.main()
