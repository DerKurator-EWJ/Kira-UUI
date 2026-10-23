"""UUI 0.8: semantic-neutral, persistent relation and dependency layer.

Identifiers and relation types are opaque strings; the engine never interprets
lexical similarity, numeric proximity, or labels as semantics. Dependencies
are explicit edges BETWEEN ASSERTIONS, not inferred from adjacency.
"""
import json
import sqlite3
import uuid

class RelationStore:
    def __init__(self, connection):
        self.db = connection
        self.db.execute("""CREATE TABLE IF NOT EXISTS uui08_nodes (
            id TEXT PRIMARY KEY, label TEXT)""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS uui08_relations (
            id TEXT PRIMARY KEY, source TEXT NOT NULL, type TEXT NOT NULL,
            target TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active',
            FOREIGN KEY(source) REFERENCES uui08_nodes(id),
            FOREIGN KEY(target) REFERENCES uui08_nodes(id))""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS uui08_dependencies (
            prerequisite TEXT NOT NULL, dependent TEXT NOT NULL,
            PRIMARY KEY(prerequisite, dependent),
            FOREIGN KEY(prerequisite) REFERENCES uui08_relations(id),
            FOREIGN KEY(dependent) REFERENCES uui08_relations(id))""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS uui08_revisions (
            id TEXT PRIMARY KEY, relation_id TEXT NOT NULL, reason TEXT NOT NULL,
            old_status TEXT NOT NULL, new_status TEXT NOT NULL,
            affected_json TEXT NOT NULL)""")
        self.db.commit()

    @staticmethod
    def ident(value):
        if not isinstance(value,str) or not value or len(value)>200:
            raise ValueError('Expected a nonempty opaque string identifier (max 200)')
        return value

    def node(self, ident, label=None):
        self.ident(ident)
        if label is not None and not isinstance(label,str):raise ValueError('label must be text')
        self.db.execute('INSERT OR IGNORE INTO uui08_nodes(id,label) VALUES (?,NULL)',(ident,))
        if label is not None:
            self.db.execute('UPDATE uui08_nodes SET label=? WHERE id=?',(label,ident))
        self.db.commit()
        return {'id':ident,'label':self.db.execute('SELECT label FROM uui08_nodes WHERE id=?',(ident,)).fetchone()[0]}

    def relation(self, source, typ, target, rid):
        for x in (source,typ,target,rid):self.ident(x)
        for node in (source,target):
            if not self.db.execute('SELECT 1 FROM uui08_nodes WHERE id=?',(node,)).fetchone():
                raise ValueError('Unknown node '+node)
        if self.db.execute('SELECT 1 FROM uui08_relations WHERE id=?',(rid,)).fetchone():
            raise ValueError('Duplicate relation ID '+rid)
        self.db.execute('INSERT INTO uui08_relations(id,source,type,target,status) VALUES (?,?,?,?,?)',
                        (rid,source,typ,target,'active'))
        self.db.commit()
        return {'id':rid,'source':source,'type':typ,'target':target,'status':'active'}

    def dependency(self, prerequisite, dependent):
        for rid in (prerequisite,dependent):
            self.ident(rid)
            if not self.db.execute('SELECT 1 FROM uui08_relations WHERE id=?',(rid,)).fetchone():
                raise ValueError('Unknown relation '+rid)
        if prerequisite==dependent:raise ValueError('Self dependency is not permitted')
        # reject cycles to ensure stable dependency traversal
        if prerequisite in self.affected(dependent):raise ValueError('Dependency cycle')
        self.db.execute('INSERT OR IGNORE INTO uui08_dependencies VALUES (?,?)',(prerequisite,dependent))
        self.db.commit()
        return {'prerequisite':prerequisite,'dependent':dependent}

    def affected(self, relation_id):
        self.ident(relation_id)
        if not self.db.execute('SELECT 1 FROM uui08_relations WHERE id=?',(relation_id,)).fetchone():
            raise ValueError('Unknown relation '+relation_id)
        seen={relation_id};frontier=[relation_id];result=[]
        while frontier:
            current=frontier.pop(0)
            for (child,) in self.db.execute('SELECT dependent FROM uui08_dependencies WHERE prerequisite=? ORDER BY dependent',(current,)):
                if child not in seen:
                    seen.add(child);result.append(child);frontier.append(child)
        return result

    def revise(self, relation_id, reason):
        self.ident(relation_id)
        if not isinstance(reason,str) or not reason.strip():raise ValueError('Revision reason required')
        row=self.db.execute('SELECT status FROM uui08_relations WHERE id=?',(relation_id,)).fetchone()
        if row is None:raise ValueError('Unknown relation '+relation_id)
        if row[0]!='active':raise ValueError('Only active relations can be revised')
        impacted=self.affected(relation_id)
        revision_id='rev:'+uuid.uuid4().hex
        with self.db:
            self.db.execute("UPDATE uui08_relations SET status='superseded' WHERE id=?",(relation_id,))
            for rid in impacted:
                self.db.execute("UPDATE uui08_relations SET status='needs_review' WHERE id=? AND status='active'",(rid,))
            self.db.execute('INSERT INTO uui08_revisions VALUES (?,?,?,?,?,?)',
                (revision_id,relation_id,reason,'active','superseded',json.dumps(impacted)))
        return {'revision_id':revision_id,'relation_id':relation_id,'status':'superseded',
                'needs_review':impacted,'reason':reason}

    def inspect(self, relation_id):
        self.ident(relation_id)
        row=self.db.execute('SELECT id,source,type,target,status FROM uui08_relations WHERE id=?',(relation_id,)).fetchone()
        if not row:raise ValueError('Unknown relation '+relation_id)
        return dict(zip(('id','source','type','target','status'),row),dependents=self.affected(relation_id))
