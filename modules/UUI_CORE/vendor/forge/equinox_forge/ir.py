from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any

class Modality(str, Enum):
    TEXT = "text"
    AUDIO = "audio"
    VISION = "vision"
    VIDEO = "video"
    CROSSMODAL = "crossmodal"
    CONTEXTUAL_USE = "contextual_use"
    CONTEXTUAL_SIMILARITY = "contextual_similarity"

class NodeKind(str, Enum):
    QUANT = "quant"
    SURFACE = "surface"
    COMPOSITE = "composite"
    CONCEPT = "concept"
    TEMPLATE = "template"

class RelationType(str, Enum):
    ASSOCIATION = "association"
    TRANSITION = "transition"
    SUBSTITUTION = "substitution"
    COMPOSITION = "composition"
    PREDICTION = "prediction"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    IS_A = "is_a"
    PROPERTY = "property"
    ACTION = "action"
    CROSSMODAL = "crossmodal"
    CONTEXTUAL_USE = "contextual_use"
    CONTEXTUAL_SIMILARITY = "contextual_similarity"

@dataclass(frozen=True)
class SourceRef:
    source_id: str
    source_version: str = ""
    extraction: str = "unknown"

@dataclass
class IRNode:
    node_id: str
    modality: str
    kind: str
    level: int
    surface: str | None = None
    members: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class RelationEvidence:
    from_id: str
    to_id: str
    relation_type: str
    score: float
    confidence: float
    source: SourceRef
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class CompiledRelation:
    from_id: str
    to_id: str
    relation_type: str
    score: float
    confidence: float
    source_count: int
    evidence_count: int

@dataclass
class ForgeDocument:
    format: str = "equinox-forge-ir-v1"
    nodes: list[IRNode] = field(default_factory=list)
    evidence: list[RelationEvidence] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "nodes": [asdict(n) for n in self.nodes],
            "evidence": [
                {
                    **{k: v for k, v in asdict(e).items() if k != "source"},
                    "source": asdict(e.source),
                }
                for e in self.evidence
            ],
        }
