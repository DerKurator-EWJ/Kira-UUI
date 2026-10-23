from __future__ import annotations
from dataclasses import dataclass
from .ir import ForgeDocument

@dataclass
class ValidationIssue:
    severity: str
    message: str


def validate_document(doc: ForgeDocument) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if doc.format != "equinox-forge-ir-v1":
        issues.append(ValidationIssue("error", f"unsupported format: {doc.format}"))
    nodes = {n.node_id: n for n in doc.nodes}
    if len(nodes) != len(doc.nodes):
        issues.append(ValidationIssue("error", "duplicate node_id entries"))
    for n in doc.nodes:
        if n.level < 0:
            issues.append(ValidationIssue("error", f"{n.node_id}: negative level"))
        if n.level == 0 and n.members:
            issues.append(ValidationIssue("error", f"{n.node_id}: Q node cannot have members"))
        if n.level > 0 and not n.members:
            issues.append(ValidationIssue("error", f"{n.node_id}: macro node requires members"))
        for m in n.members:
            if m not in nodes:
                issues.append(ValidationIssue("error", f"{n.node_id}: missing member {m}"))
        child_levels = [nodes[m].level for m in n.members if m in nodes]
        if child_levels and n.level != max(child_levels) + 1:
            issues.append(ValidationIssue("error", f"{n.node_id}: level {n.level} != max(child)+1 ({max(child_levels)+1})"))
    for e in doc.evidence:
        if e.from_id not in nodes or e.to_id not in nodes:
            issues.append(ValidationIssue("error", f"relation references unknown node: {e.from_id}->{e.to_id}"))
        if e.from_id == e.to_id:
            issues.append(ValidationIssue("warning", f"self relation: {e.from_id}"))
        if not (0.0 <= e.score <= 1.0):
            issues.append(ValidationIssue("error", f"relation score outside [0,1]: {e.score}"))
        if not (0.0 <= e.confidence <= 1.0):
            issues.append(ValidationIssue("error", f"relation confidence outside [0,1]: {e.confidence}"))
    # Cycle check only on composition edges.
    graph = {n.node_id: list(n.members) for n in doc.nodes}
    visiting: set[str] = set(); visited: set[str] = set()
    def dfs(x: str) -> bool:
        if x in visiting: return True
        if x in visited: return False
        visiting.add(x)
        for y in graph.get(x, []):
            if dfs(y): return True
        visiting.remove(x); visited.add(x); return False
    for x in graph:
        if dfs(x):
            issues.append(ValidationIssue("error", "composition graph contains a cycle"))
            break
    return issues
