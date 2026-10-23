"""UUI 0.9: exact, typed two-edge chain motifs, no inferred semantics.

Motifs are descriptive snapshots. They are not automatically asserted as truths,
and only active edges participate in new detections.
"""
import hashlib
import json

class PatternStore:
    def __init__(self, relation_store):
        self.relations = relation_store
        self.db = relation_store.db
        self.db.execute("""CREATE TABLE IF NOT EXISTS uui09_macros (
            id TEXT PRIMARY KEY, type_left TEXT NOT NULL, type_right TEXT NOT NULL,
            instance_count INTEGER NOT NULL, created_from TEXT NOT NULL)""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS uui09_instances (
            macro_id TEXT NOT NULL, left_relation TEXT NOT NULL,
            right_relation TEXT NOT NULL, middle_node TEXT NOT NULL,
            PRIMARY KEY(macro_id,left_relation,right_relation))""")
        self.db.commit()

    def detect(self, min_instances=2):
        if type(min_instances) is not int or not 2 <= min_instances <= 100000:
            raise ValueError('min_instances must be an integer between 2 and 100000')
        rows=self.db.execute("""SELECT a.id,b.id,a.type,b.type,a.target
            FROM uui08_relations a JOIN uui08_relations b ON a.target=b.source
            WHERE a.status='active' AND b.status='active' AND a.id<>b.id
              AND a.source<>b.target
            ORDER BY a.type,b.type,a.id,b.id""").fetchall()
        groups={}
        for left,right,t1,t2,middle in rows:
            groups.setdefault((t1,t2),[]).append((left,right,middle))
        results=[]
        for (t1,t2),instances in sorted(groups.items()):
            if len(instances)<min_instances:continue
            # Opaque type IDs are compared for exact equality, not lexical meaning.
            fingerprint=hashlib.sha256(json.dumps([t1,t2],ensure_ascii=False).encode()).hexdigest()[:24]
            results.append({'pattern_id':'pattern:'+fingerprint,'type_left':t1,
                            'type_right':t2,'count':len(instances),'instances':
                            [{'left':a,'right':b,'middle':c} for a,b,c in instances]})
        return results

    def consolidate(self, min_instances=2):
        patterns=self.detect(min_instances)
        with self.db:
            for item in patterns:
                self.db.execute("INSERT INTO uui09_macros(id,type_left,type_right,instance_count,created_from) VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET instance_count=excluded.instance_count,created_from=excluded.created_from",
                    (item['pattern_id'],item['type_left'],item['type_right'],item['count'],'exact_two_edge_chain'))
                self.db.execute('DELETE FROM uui09_instances WHERE macro_id=?',(item['pattern_id'],))
                self.db.executemany('INSERT INTO uui09_instances VALUES(?,?,?,?)',
                    [(item['pattern_id'],x['left'],x['right'],x['middle']) for x in item['instances']])
        return [{'id':p['pattern_id'],'count':p['count']} for p in patterns]

    def expand(self, macro_id):
        if not isinstance(macro_id,str):raise ValueError('macro id must be text')
        row=self.db.execute('SELECT type_left,type_right,instance_count FROM uui09_macros WHERE id=?',(macro_id,)).fetchone()
        if row is None:raise ValueError('Unknown macro '+macro_id)
        entries=self.db.execute("SELECT i.left_relation,i.right_relation,i.middle_node,a.status,b.status FROM uui09_instances i JOIN uui08_relations a ON a.id=i.left_relation JOIN uui08_relations b ON b.id=i.right_relation WHERE i.macro_id=? ORDER BY i.left_relation,i.right_relation",(macro_id,)).fetchall()
        return {'id':macro_id,'types':list(row[:2]),'stored_count':row[2],
                'instances':[{'left':a,'right':b,'middle':c,'currently_active':s1=='active' and s2=='active'}
                             for a,b,c,s1,s2 in entries]}
