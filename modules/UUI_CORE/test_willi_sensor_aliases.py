import json
import unittest
from uui_bridge import UUIBridge
from uui_lang import Runtime
from uui_cognition import WILLI_SENSOR_ALIASES, SENSOR_OPS

class WilliSensorAliasTests(unittest.TestCase):
    def test_registered(self):
        self.assertTrue(set(WILLI_SENSOR_ALIASES).issubset(SENSOR_OPS))
    def test_single_and_inspect(self):
        b=UUIBridge()
        try:
            payload=dict(event_id="w1", modality="visual", source="cam", sequence=1, prototype_id=7, features=[0.2]*7)
            src="hier auge_auf("+json.dumps(json.dumps(payload))+")\nhier was_war(\"w1\")"
            result=Runtime(b).run(src)
            self.assertEqual(len(result),2)
            self.assertIn('prototype_node',result[0])
            self.assertIn('w1',result[1])
        finally:b.close()
    def test_batch(self):
        b=UUIBridge()
        try:
            payload=[dict(event_id="b"+str(i),modality="audio",source="mic",sequence=i,prototype_id=7,features=[0.2]*6) for i in range(2)]
            src="hier nase_schnueffeln("+json.dumps(json.dumps(payload))+")"
            result=Runtime(b).run(src)
            self.assertEqual(len(result),1)
            self.assertIsNotNone(b.sensors.inspect('b1'))
        finally:b.close()
if __name__=="__main__":unittest.main()
