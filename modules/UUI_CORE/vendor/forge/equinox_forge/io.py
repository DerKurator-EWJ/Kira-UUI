from __future__ import annotations
import json
from pathlib import Path
from .ir import ForgeDocument, IRNode, RelationEvidence, SourceRef

def load_document(path: str | Path) -> ForgeDocument:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = [IRNode(**n) for n in raw.get("nodes",[])]
    evidence = []
    for e in raw.get("evidence",[]):
        e = dict(e); src = SourceRef(**e.pop("source")); evidence.append(RelationEvidence(source=src,**e))
    return ForgeDocument(format=raw.get("format",""),nodes=nodes,evidence=evidence)

def save_document(doc: ForgeDocument, path: str | Path):
    Path(path).write_text(json.dumps(doc.to_dict(),ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
