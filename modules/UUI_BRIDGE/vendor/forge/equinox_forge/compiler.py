from __future__ import annotations
from dataclasses import dataclass
import math
import sqlite3
from .ir import CompiledRelation
from .source_lineage import SourceLineageGraph


@dataclass
class CompilePolicy:
    min_score: float = 0.05
    min_confidence: float = 0.05
    min_sources: int = 1


def _legacy_aggregate(evs: list[tuple[float, float, str]]) -> tuple[float, float, int]:
    """Original pre-lineage aggregation, retained when no lineage metadata applies."""
    conf_sum = sum(c for _, c, _ in evs)
    if conf_sum <= 0:
        return 0.0, 0.0, 0
    score = sum(s * c for s, c, _ in evs) / conf_sum
    confidence = 1.0 - math.prod(1.0 - c for _, c, _ in evs)
    source_count = len({src for _, _, src in evs})
    return score, confidence, source_count


def _lineage_aggregate(
    evs: list[tuple[float, float, str]], graph: SourceLineageGraph
) -> tuple[float, float, int]:
    """Aggregate correlated evidence once per independent lineage component.

    Within one component, evidence can refine the component score but cannot
    manufacture independent confidence. Component confidence is therefore the
    strongest confidence observed inside that correlated family. Across
    independent components, confidence combines with noisy-OR as before.
    """
    source_keys = sorted({src for _, _, src in evs})
    components = graph.components(source_keys)
    component_rows: list[tuple[float, float]] = []
    for component in components:
        members = set(component.sources)
        per_source: list[tuple[float, float]] = []
        for source in sorted(members):
            rows = [(s, c) for s, c, src in evs if src == source]
            den = sum(c for _, c in rows)
            if den <= 0:
                continue
            source_score = sum(s * c for s, c in rows) / den
            source_confidence = max(c for _, c in rows)
            per_source.append((source_score, source_confidence))
        if not per_source:
            continue
        # The component estimate must not depend on how many aliases were minted
        # from the same ancestry.  The midrange is deliberately conservative and
        # count-invariant: duplicating either endpoint cannot move it.
        source_scores = [s for s, _ in per_source]
        score = (min(source_scores) + max(source_scores)) / 2.0
        confidence = max(c for _, c in per_source)
        component_rows.append((score, confidence))
    if not component_rows:
        return 0.0, 0.0, 0
    den = sum(c for _, c in component_rows)
    score = sum(s * c for s, c in component_rows) / den if den > 0 else 0.0
    confidence = 1.0 - math.prod(1.0 - c for _, c in component_rows)
    return score, confidence, len(component_rows)


def aggregate_relations(db: sqlite3.Connection, policy: CompilePolicy = CompilePolicy()) -> list[CompiledRelation]:
    rows = db.execute("""
      SELECT from_id,to_id,relation_type,score,confidence,source_id,source_version
      FROM evidence
      ORDER BY from_id,to_id,relation_type,evidence_id
    """).fetchall()
    buckets: dict[tuple[str,str,str], list[tuple[float,float,str]]] = {}
    for a,b,t,s,c,src,ver in rows:
        buckets.setdefault((a,b,t),[]).append((float(s),float(c),f"{src}@{ver}"))

    graph = SourceLineageGraph.from_db(db)
    out: list[CompiledRelation] = []
    for (a,b,t), evs in buckets.items():
        source_keys = {src for _, _, src in evs}
        # Backward compatibility: an old database with no ancestry registered for
        # any participating source keeps byte-for-byte aggregation semantics.
        use_lineage = any(src in graph.parents for src in source_keys)
        if use_lineage:
            score, confidence, source_count = _lineage_aggregate(evs, graph)
        else:
            score, confidence, source_count = _legacy_aggregate(evs)
        if source_count <= 0:
            continue
        if score < policy.min_score or confidence < policy.min_confidence or source_count < policy.min_sources:
            continue
        out.append(CompiledRelation(a,b,t,score,confidence,source_count,len(evs)))
    return out
