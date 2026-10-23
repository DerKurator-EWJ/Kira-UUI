from pathlib import Path
import sys, sqlite3, tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from equinox_forge.database import CanonicalDB
from equinox_forge.ir import RelationEvidence, SourceRef

OLD_SCHEMA='''
CREATE TABLE evidence(
  evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
  from_id TEXT NOT NULL,to_id TEXT NOT NULL,relation_type TEXT NOT NULL,
  score REAL NOT NULL,confidence REAL NOT NULL,source_id TEXT NOT NULL,
  source_version TEXT NOT NULL,extraction TEXT NOT NULL,metadata_json TEXT NOT NULL DEFAULT '{}'
);
'''

def run():
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'old.sqlite'
        db=sqlite3.connect(p); db.executescript(OLD_SCHEMA); db.close()
        with CanonicalDB(p) as c:
            cols={r[1] for r in c.db.execute('PRAGMA table_info(evidence)')}
            assert 'evidence_fingerprint' in cols
            e=RelationEvidence('a','b','association',0.7,0.5,SourceRef('s','1','x'))
            c.add_evidence(e); c.add_evidence(e); c.commit()
            assert c.db.execute('SELECT COUNT(*) FROM evidence').fetchone()[0] == 1
    print('FORGE_V03_DB_MIGRATION_OK')
if __name__=='__main__': run()
