import json
import tempfile
import unittest
from pathlib import Path
from uui_bridge import UUIBridge
from uui_lang import Runtime

class PatternTests(unittest.TestCase):
    def seed(self,b):
        for n in 'ABCDEF':b.relations.node(n)
        for x in [('A','t1','B','r1'),('B','t2','C','r2'),('D','t1','E','r3'),('E','t2','F','r4')]:b.relations.relation(*x)
    def test_detect_and_expand(self):
        b=UUIBridge()
        try:
            self.seed(b)
            patterns=b.patterns.detect()
            self.assertEqual(len(patterns),1)
            self.assertEqual(patterns[0]['count'],2)
            mid=b.patterns.consolidate()[0]['id']
            self.assertEqual(len(b.patterns.expand(mid)['instances']),2)
            b.relations.revise('r1','invalidated')
            self.assertEqual(len(b.patterns.detect()),0)
            self.assertEqual(sum(i['currently_active'] for i in b.patterns.expand(mid)['instances']),1)
        finally:b.close()
    def test_no_false_match_for_types_or_adjacency(self):
        b=UUIBridge()
        try:
            self.seed(b)
            b.relations.revise('r3','bad')
            self.assertEqual(b.patterns.detect(),[])
        finally:b.close()
    def test_persistence(self):
        with tempfile.TemporaryDirectory() as d:
            path=str(Path(d)/'test.sqlite')
            b=UUIBridge(kira_db=path)
            self.seed(b)
            mid=b.patterns.consolidate()[0]['id']
            b.close()
            b=UUIBridge(kira_db=path)
            try:self.assertEqual(b.patterns.expand(mid)['stored_count'],2)
            finally:b.close()
    def test_uui_integration(self):
        b=UUIBridge()
        try:
            result=Runtime(b).run(Path(__file__).with_name('pattern_demo.uui').read_text())
            self.assertEqual(json.loads(result[-1])[0]['count'],2)
        finally:b.close()
    def test_no_overinterpretation(self):
        b=UUIBridge()
        try:
            self.seed(b)
            self.assertEqual(b.patterns.detect(3),[])
            with self.assertRaises(ValueError):b.patterns.detect(1)
        finally:b.close()

if __name__=='__main__':unittest.main()
