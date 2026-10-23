import unittest
import tempfile
import os
import json
from uui_bridge import UUIBridge
from uui_lang import Runtime

class SensorTests(unittest.TestCase):
    def event(self,eid='e1',seq=1,proto=-50000001,modality='visual'):
        return dict(event_id=eid,modality=modality,source='cam0' if modality=='visual' else 'mic0',sequence=seq,prototype_id=proto,features=[0.2]*(7 if modality=='visual' else 6))
    def test_persistent_and_relational(self):
        with tempfile.TemporaryDirectory() as d:
            path=os.path.join(d,'k.sqlite')
            b=UUIBridge(kira_db=path)
            b.sensors.ingest(self.event())
            b.sensors.ingest(self.event('e2',2))
            self.assertEqual(b.sensors.inspect('e1')['prototype_id'],str(-50000001))
            self.assertEqual(b.relations.inspect('sensor.next:e2')['type'],'sensor.precedes')
            b.close()
            b=UUIBridge(kira_db=path)
            self.assertEqual(b.sensors.inspect('e2')['sequence'],2)
            self.assertTrue(len(b.patterns.detect(2))>=1)
            b.close()
    def test_no_semantics_or_numeric_proximity(self):
        b=UUIBridge()
        b.sensors.ingest(self.event())
        b.sensors.ingest(self.event('e2',2,-50000002))
        self.assertNotEqual(b.sensors.inspect('e1')['prototype_id'],b.sensors.inspect('e2')['prototype_id'])
        b.close()
    def test_validation_and_dedup(self):
        b=UUIBridge();b.sensors.ingest(self.event())
        with self.assertRaises(ValueError):b.sensors.ingest(self.event())
        with self.assertRaises(ValueError):b.sensors.ingest(self.event('e2',2,modality='audio')|{'features':[float('nan')]*6})
        with self.assertRaises(ValueError):b.sensors.ingest(self.event('e2',1))
        b.close()
    def test_uui_command(self):
        b=UUIBridge()
        payload=json.dumps(self.event())
        source='hier sensor_rein('+json.dumps(payload)+')\nhier sensor_nachsehen("e1")'
        result=Runtime(b).run(source)
        self.assertEqual(len(result),2)
        self.assertIn('prototype_node',result[0]);b.close()
    def test_batch_and_macro(self):
        b=UUIBridge()
        for i in range(4):b.sensors.ingest(self.event('e'+str(i),i))
        # Produced -> event -> prototype repeats across four events.
        patterns=b.patterns.detect(2)
        self.assertTrue(any(p['type_left']=='sensor.produced' and p['type_right']=='sensor.quantized_as' for p in patterns))
        b.close()
if __name__=='__main__':unittest.main()
