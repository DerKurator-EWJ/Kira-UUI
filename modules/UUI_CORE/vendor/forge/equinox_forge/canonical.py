from __future__ import annotations
import hashlib
import json
import unicodedata
from .ir import IRNode, Modality, NodeKind

TIER_NAMES = {
    0: "Q", 1: "F", 2: "E", 3: "D", 4: "C", 5: "B", 6: "A",
    7: "S", 8: "SS", 9: "SSS", 10: "Ω",
}

def tier_name(level: int) -> str:
    if level <= 10:
        return TIER_NAMES.get(level, "Q")
    return f"Ω+{level-10}"

def normalize_surface(text: str) -> str:
    # Preserve spelling/case semantically; only canonicalize Unicode representation.
    return unicodedata.normalize("NFC", text)

def _digest(payload: object, prefix: str) -> str:
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(raw).hexdigest()[:24]}"

def quant_id(modality: str, symbol: str) -> str:
    symbol = normalize_surface(symbol)
    return _digest({"m": modality, "k": "quant", "s": symbol}, "q")

def surface_id(modality: str, surface: str, members: list[str]) -> str:
    return _digest({"m": modality, "k": "surface", "s": normalize_surface(surface), "members": members}, "m")

def composite_id(modality: str, kind: str, members: list[str], label: str | None = None) -> str:
    return _digest({"m": modality, "k": kind, "members": members, "label": label or ""}, "m")

def make_text_quant(ch: str) -> IRNode:
    s = normalize_surface(ch)
    return IRNode(quant_id(Modality.TEXT.value, s), Modality.TEXT.value, NodeKind.QUANT.value, 0, surface=s)

def make_text_surface(text: str) -> tuple[list[IRNode], IRNode]:
    text = normalize_surface(text)
    quants = [make_text_quant(ch) for ch in text]
    members = [q.node_id for q in quants]
    node = IRNode(surface_id(Modality.TEXT.value, text, members), Modality.TEXT.value, NodeKind.SURFACE.value, 1, surface=text, members=members)
    # local exact dedupe, global dedupe happens in the DB
    uniq = {q.node_id: q for q in quants}
    return list(uniq.values()), node

def make_composite(modality: str, members: list[IRNode], kind: str = NodeKind.COMPOSITE.value, label: str | None = None) -> IRNode:
    if not members:
        raise ValueError("composite requires members")
    level = max(m.level for m in members) + 1
    return IRNode(composite_id(modality, kind, [m.node_id for m in members], label), modality, kind, level, surface=label, members=[m.node_id for m in members])
