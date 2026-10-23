import json
import tempfile
import unittest
from pathlib import Path
from uui_bridge import UUIBridge
from uui_lang import Runtime, UUIError

class NeutralRelationTests(unittest.TestCase):
    def test_transitive_and_persistent(self):
        with tempfile.TemporaryDirectory() as folder:
            db=str(Path(folder)/'data.sqlite')
            bridge=UUIBridge(kira_db=db)
            try:
                r=bridge.relations
                for n in ('A','B','C','D'):r.node(n)
                r.relation('A','opaque:1','B','r1')
                r.relation('B','opaque:2','C','r2')
                r.relation('C','opaque:3','D','r3')
                r.dependency('r1','r2');r.dependency('r2','r3')
                self.assertEqual(r.affected('r1'),['r2','r3'])
                self.assertRaises(ValueError,r.dependency,'r3','r1')
                change=r.revise('r1','changed premise')
                self.assertEqual(change['needs_review'],['r2','r3'])
            finally:bridge.close()
            bridge=UUIBridge(kira_db=db)
            try:
                self.assertEqual(bridge.relations.inspect('r1')['status'],'superseded')
                self.assertEqual(bridge.relations.inspect('r3')['status'],'needs_review')
                self.assertEqual(bridge.relations.affected('r1'),['r2','r3'])
            finally:bridge.close()

    def test_connection_is_not_dependency(self):
        b=UUIBridge()
        try:
            r=b.relations
            for n in ('1','2','3'):r.node(n)
            r.relation('1','t','2','r1');r.relation('2','t','3','r2')
            self.assertEqual(r.affected('r1'),[])
            r.revise('r1','explicit')
            self.assertEqual(r.inspect('r2')['status'],'active')
        finally:b.close()

    def test_uui_commands_and_label_non_semantics(self):
        b=UUIBridge()
        try:
            source='hier knoten("A")\nhier knoten("B")\nhier benenne("A", "Apfel")\nhier relation("A", "opaque:42", "B", "r1")\nhier folgen_von("r1")\nhier pruef_relation("r1")\n'
            result=Runtime(b).run(source)
            self.assertEqual(json.loads(result[-2])['affected'],[])
            self.assertEqual(json.loads(result[-1])['type'],'opaque:42')
            self.assertEqual(b.relations.inspect('r1')['source'],'A')
        finally:b.close()

    def test_duplicate_and_missing_ids(self):
        b=UUIBridge()
        try:
            r=b.relations;r.node('A');r.node('B');r.relation('A','t','B','r')
            self.assertRaises(ValueError,r.relation,'A','t','B','r')
            self.assertRaises(ValueError,r.relation,'A','t','missing','r2')
            self.assertRaises(ValueError,r.dependency,'r','unknown')
        finally:b.close()

if __name__=='__main__':unittest.main()
