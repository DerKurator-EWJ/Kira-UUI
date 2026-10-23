from __future__ import annotations
import json
import sqlite3
import hashlib
from pathlib import Path
from .ir import IRNode, RelationEvidence

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS nodes(
  node_id TEXT PRIMARY KEY,
  modality TEXT NOT NULL,
  kind TEXT NOT NULL,
  level INTEGER NOT NULL,
  surface TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS node_members(
  parent_id TEXT NOT NULL,
  position INTEGER NOT NULL,
  child_id TEXT NOT NULL,
  PRIMARY KEY(parent_id, position),
  FOREIGN KEY(parent_id) REFERENCES nodes(node_id),
  FOREIGN KEY(child_id) REFERENCES nodes(node_id)
);
CREATE TABLE IF NOT EXISTS evidence(
  evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
  evidence_fingerprint TEXT UNIQUE,
  from_id TEXT NOT NULL,
  to_id TEXT NOT NULL,
  relation_type TEXT NOT NULL,
  score REAL NOT NULL,
  confidence REAL NOT NULL,
  source_id TEXT NOT NULL,
  source_version TEXT NOT NULL,
  extraction TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_evidence_relation ON evidence(from_id,to_id,relation_type);
CREATE INDEX IF NOT EXISTS idx_evidence_source ON evidence(source_id,source_version);
CREATE TABLE IF NOT EXISTS source_lineage_edges(
  child_source_id TEXT NOT NULL,
  child_source_version TEXT NOT NULL,
  parent_source_id TEXT NOT NULL,
  parent_source_version TEXT NOT NULL,
  relation TEXT NOT NULL DEFAULT 'derived_from',
  metadata_json TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY(child_source_id,child_source_version,parent_source_id,parent_source_version,relation)
);
CREATE INDEX IF NOT EXISTS idx_source_lineage_parent ON source_lineage_edges(parent_source_id,parent_source_version);
CREATE INDEX IF NOT EXISTS idx_source_lineage_child ON source_lineage_edges(child_source_id,child_source_version);
CREATE TABLE IF NOT EXISTS source_models(
  source_id TEXT NOT NULL,
  source_version TEXT NOT NULL,
  adapter_id TEXT NOT NULL,
  adapter_hash TEXT NOT NULL,
  model_type TEXT NOT NULL DEFAULT '',
  manifest_json TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY(source_id, source_version)
);
CREATE TABLE IF NOT EXISTS source_aliases(
  source_id TEXT NOT NULL,
  source_version TEXT NOT NULL,
  external_id TEXT NOT NULL,
  node_id TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY(source_id,source_version,external_id)
);
CREATE TABLE IF NOT EXISTS hypothesis_runs(
  run_id INTEGER PRIMARY KEY AUTOINCREMENT,
  compiler TEXT NOT NULL,
  target_id TEXT NOT NULL,
  policy_hash TEXT NOT NULL,
  output_hash TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_hypothesis_runs_target ON hypothesis_runs(target_id,policy_hash,run_id);
CREATE TABLE IF NOT EXISTS hypothesis_states(
  hypothesis_id TEXT NOT NULL,
  target_id TEXT NOT NULL,
  policy_hash TEXT NOT NULL,
  membership_hash TEXT NOT NULL,
  status TEXT NOT NULL,
  first_seen_run INTEGER NOT NULL,
  last_seen_run INTEGER NOT NULL,
  seen_runs INTEGER NOT NULL DEFAULT 1,
  consecutive_seen INTEGER NOT NULL DEFAULT 1,
  missed_runs INTEGER NOT NULL DEFAULT 0,
  consensus_score REAL NOT NULL DEFAULT 0.0,
  support_sources INTEGER NOT NULL DEFAULT 0,
  conflict_sources INTEGER NOT NULL DEFAULT 0,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY(hypothesis_id,policy_hash)
);
CREATE INDEX IF NOT EXISTS idx_hypothesis_states_target ON hypothesis_states(target_id,policy_hash,status);

CREATE TABLE IF NOT EXISTS source_policies(
  source_id TEXT NOT NULL,
  source_version TEXT NOT NULL,
  enabled INTEGER NOT NULL DEFAULT 1,
  trust_weight REAL NOT NULL DEFAULT 1.0,
  note TEXT NOT NULL DEFAULT '',
  metadata_json TEXT NOT NULL DEFAULT '{}',
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(source_id,source_version)
);
CREATE TABLE IF NOT EXISTS rebuild_runs(
  rebuild_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
  target_id TEXT NOT NULL,
  source_policy_hash TEXT NOT NULL,
  output_hash TEXT NOT NULL,
  raw_evidence_count INTEGER NOT NULL,
  effective_evidence_count INTEGER NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_rebuild_runs_target ON rebuild_runs(target_id,rebuild_run_id);

CREATE TABLE IF NOT EXISTS dependency_edges(
  dependent_type TEXT NOT NULL,
  dependent_id TEXT NOT NULL,
  dependency_type TEXT NOT NULL,
  dependency_id TEXT NOT NULL,
  dependency_kind TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY(dependent_type,dependent_id,dependency_type,dependency_id,dependency_kind)
);
CREATE INDEX IF NOT EXISTS idx_dependency_lookup ON dependency_edges(dependency_type,dependency_id,dependent_type,dependent_id);
CREATE TABLE IF NOT EXISTS build_targets(
  target_id TEXT PRIMARY KEY,
  builder TEXT NOT NULL,
  policy_json TEXT NOT NULL DEFAULT '{}',
  active INTEGER NOT NULL DEFAULT 1,
  last_output_hash TEXT NOT NULL DEFAULT '',
  metadata_json TEXT NOT NULL DEFAULT '{}',
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS incremental_rebuild_runs(
  incremental_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
  change_set_hash TEXT NOT NULL,
  source_policy_hash TEXT NOT NULL,
  roots_json TEXT NOT NULL,
  affected_json TEXT NOT NULL,
  rebuilt_json TEXT NOT NULL,
  deferred_json TEXT NOT NULL,
  output_hash TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);
"""

class CanonicalDB:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)
        self._migrate_schema()
        from .artifact_store import ensure_artifact_schema
        ensure_artifact_schema(self.db)
        from .macro_discovery import ensure_discovery_schema
        ensure_discovery_schema(self.db)
        from .runtime_compression import ensure_runtime_compression_schema
        ensure_runtime_compression_schema(self.db)
        from .runtime_pack import ensure_runtime_pack_schema
        ensure_runtime_pack_schema(self.db)
        from .exception_consensus import ensure_exception_consensus_schema
        ensure_exception_consensus_schema(self.db)
        from .composition_lifecycle import ensure_composition_lifecycle_schema
        ensure_composition_lifecycle_schema(self.db)
        from .canonical_id_space import ensure_canonical_id_schema
        ensure_canonical_id_schema(self.db)

    def _migrate_schema(self):
        # v0.3 makes repeated extraction runs idempotent. Existing v0.1/v0.2 DBs
        # did not have this column, so add it non-destructively when needed.
        cols = {row[1] for row in self.db.execute("PRAGMA table_info(evidence)")}
        if "evidence_fingerprint" not in cols:
            self.db.execute("ALTER TABLE evidence ADD COLUMN evidence_fingerprint TEXT")
        self.db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_evidence_fingerprint ON evidence(evidence_fingerprint)")
        alias_cols = {row[1] for row in self.db.execute("PRAGMA table_info(source_aliases)")}
        if "metadata_json" not in alias_cols:
            self.db.execute("ALTER TABLE source_aliases ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'")

    def close(self): self.db.close()
    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    def upsert_node(self, node: IRNode):
        self.db.execute(
            "INSERT OR IGNORE INTO nodes(node_id,modality,kind,level,surface,metadata_json) VALUES(?,?,?,?,?,?)",
            (node.node_id,node.modality,node.kind,node.level,node.surface,json.dumps(node.metadata,ensure_ascii=False,sort_keys=True)),
        )
        existing = self.db.execute("SELECT modality,kind,level,surface FROM nodes WHERE node_id=?",(node.node_id,)).fetchone()
        expected = (node.modality,node.kind,node.level,node.surface)
        if tuple(existing) != expected:
            raise ValueError(f"canonical collision for {node.node_id}: {existing} != {expected}")
        for pos, child in enumerate(node.members):
            self.db.execute("INSERT OR IGNORE INTO node_members(parent_id,position,child_id) VALUES(?,?,?)",(node.node_id,pos,child))

    def add_evidence(self, e: RelationEvidence):
        metadata_json = json.dumps(e.metadata,ensure_ascii=False,sort_keys=True)
        raw = json.dumps({
            "from":e.from_id,"to":e.to_id,"type":e.relation_type,
            "score":float(e.score),"confidence":float(e.confidence),
            "source":e.source.source_id,"version":e.source.source_version,
            "extraction":e.source.extraction,"metadata":e.metadata,
        },ensure_ascii=False,separators=(",",":"),sort_keys=True).encode("utf-8")
        fp = hashlib.sha256(raw).hexdigest()
        self.db.execute(
            "INSERT OR IGNORE INTO evidence(evidence_fingerprint,from_id,to_id,relation_type,score,confidence,source_id,source_version,extraction,metadata_json) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (fp,e.from_id,e.to_id,e.relation_type,float(e.score),float(e.confidence),e.source.source_id,e.source.source_version,e.source.extraction,metadata_json),
        )

    def register_source_model(self, source_id: str, source_version: str, adapter_id: str, adapter_hash: str, model_type: str, manifest: dict):
        self.db.execute(
            "INSERT OR REPLACE INTO source_models(source_id,source_version,adapter_id,adapter_hash,model_type,manifest_json) VALUES(?,?,?,?,?,?)",
            (source_id,source_version,adapter_id,adapter_hash,model_type,json.dumps(manifest,ensure_ascii=False,sort_keys=True)),
        )

    def alias(self, source_id: str, source_version: str, external_id: str, node_id: str, metadata: dict | None = None):
        self.db.execute(
            "INSERT OR REPLACE INTO source_aliases(source_id,source_version,external_id,node_id,metadata_json) VALUES(?,?,?,?,?)",
            (source_id,source_version,external_id,node_id,json.dumps(metadata or {},ensure_ascii=False,sort_keys=True)),
        )

    def commit(self): self.db.commit()

    def stats(self):
        q = lambda sql: self.db.execute(sql).fetchone()[0]
        return {"nodes":q("SELECT COUNT(*) FROM nodes"),"evidence":q("SELECT COUNT(*) FROM evidence"),"sources":q("SELECT COUNT(DISTINCT source_id||'@'||source_version) FROM evidence"),"source_lineage_edges":q("SELECT COUNT(*) FROM source_lineage_edges"),"registered_models":q("SELECT COUNT(*) FROM source_models"),"hypothesis_runs":q("SELECT COUNT(*) FROM hypothesis_runs"),"hypotheses":q("SELECT COUNT(*) FROM hypothesis_states"),"source_policies":q("SELECT COUNT(*) FROM source_policies"),"rebuild_runs":q("SELECT COUNT(*) FROM rebuild_runs"),"dependency_edges":q("SELECT COUNT(*) FROM dependency_edges"),"build_targets":q("SELECT COUNT(*) FROM build_targets"),"incremental_rebuild_runs":q("SELECT COUNT(*) FROM incremental_rebuild_runs"),"derived_artifacts":q("SELECT COUNT(*) FROM derived_artifacts"),"derived_artifact_versions":q("SELECT COUNT(*) FROM derived_artifact_versions"),"macro_discovery_runs":q("SELECT COUNT(*) FROM macro_discovery_runs"),"macro_candidates":q("SELECT COUNT(*) FROM macro_candidates"),"runtime_compression_runs":q("SELECT COUNT(*) FROM runtime_compression_runs"),"runtime_promoted_macros":q("SELECT COUNT(*) FROM runtime_promoted_macros"),"runtime_parent_rewrites":q("SELECT COUNT(*) FROM runtime_parent_rewrites"),"runtime_base_pack_runs":q("SELECT COUNT(*) FROM runtime_base_pack_runs"),"exception_hypothesis_runs":q("SELECT COUNT(*) FROM exception_hypothesis_runs"),"exception_hypotheses":q("SELECT COUNT(*) FROM exception_hypothesis_states"),"composition_hypothesis_runs":q("SELECT COUNT(*) FROM composition_hypothesis_runs"),"composition_hypotheses":q("SELECT COUNT(*) FROM composition_hypothesis_states")}
