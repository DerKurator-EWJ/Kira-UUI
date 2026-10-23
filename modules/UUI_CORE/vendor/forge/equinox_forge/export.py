from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from .compiler import aggregate_relations
from .canonical import tier_name


def export_runtime_snapshot(db: sqlite3.Connection, out_path: str | Path, pack_id: str = "forge-preview", version: int = 1):
    """Forge-native preview runtime format. Not yet the Android Step-46 EQKI bridge for text Qs."""
    nodes = []
    for node_id,modality,kind,level,surface,metadata in db.execute("SELECT node_id,modality,kind,level,surface,metadata_json FROM nodes ORDER BY level,node_id"):
        members = [r[0] for r in db.execute("SELECT child_id FROM node_members WHERE parent_id=? ORDER BY position",(node_id,))]
        nodes.append({"id":node_id,"modality":modality,"kind":kind,"level":level,"tier":tier_name(level),"surface":surface,"members":members,"metadata":json.loads(metadata)})
    relations = [r.__dict__ for r in aggregate_relations(db)]
    payload = {"format":"equinox-forge-runtime-preview-v1","pack":{"id":pack_id,"version":version},"nodes":nodes,"relations":relations}
    Path(out_path).write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
