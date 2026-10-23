from __future__ import annotations

from dataclasses import dataclass
import json
import sqlite3
from typing import Iterable

SOURCE_LINEAGE_VERSION = "0.37"


def source_key(source_id: str, source_version: str) -> str:
    return f"{source_id}@{source_version}"


def split_source_key(key: str) -> tuple[str, str]:
    if "@" not in key:
        return key, ""
    return tuple(key.rsplit("@", 1))  # type: ignore[return-value]


def ensure_source_lineage_schema(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS source_lineage_edges(
          child_source_id TEXT NOT NULL,
          child_source_version TEXT NOT NULL,
          parent_source_id TEXT NOT NULL,
          parent_source_version TEXT NOT NULL,
          relation TEXT NOT NULL DEFAULT 'derived_from',
          metadata_json TEXT NOT NULL DEFAULT '{}',
          PRIMARY KEY(
            child_source_id, child_source_version,
            parent_source_id, parent_source_version, relation
          )
        );
        CREATE INDEX IF NOT EXISTS idx_source_lineage_parent
          ON source_lineage_edges(parent_source_id,parent_source_version);
        CREATE INDEX IF NOT EXISTS idx_source_lineage_child
          ON source_lineage_edges(child_source_id,child_source_version);
        """
    )


def register_source_lineage(
    db: sqlite3.Connection,
    child_source_id: str,
    child_source_version: str,
    parent_source_id: str,
    parent_source_version: str,
    *,
    relation: str = "derived_from",
    metadata: dict | None = None,
) -> None:
    ensure_source_lineage_schema(db)
    if source_key(child_source_id, child_source_version) == source_key(parent_source_id, parent_source_version):
        raise ValueError("a source cannot be its own lineage parent")
    db.execute(
        """
        INSERT OR REPLACE INTO source_lineage_edges(
          child_source_id,child_source_version,parent_source_id,parent_source_version,relation,metadata_json
        ) VALUES(?,?,?,?,?,?)
        """,
        (
            child_source_id,
            child_source_version,
            parent_source_id,
            parent_source_version,
            relation,
            json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
        ),
    )
    db.commit()


@dataclass(frozen=True)
class LineageComponent:
    sources: tuple[str, ...]
    roots: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"sources": list(self.sources), "roots": list(self.roots)}


class SourceLineageGraph:
    """Conservative ancestry graph for epistemic source independence.

    A source with no registered parent is its own root. Two observed sources are
    treated as correlated when their transitive root sets overlap. Components are
    therefore connected by shared ancestry, including multi-parent derivations.
    """

    def __init__(self, parents: dict[str, set[str]] | None = None):
        self.parents: dict[str, set[str]] = {k: set(v) for k, v in (parents or {}).items()}
        self._root_cache: dict[str, frozenset[str]] = {}

    @classmethod
    def from_db(cls, db: sqlite3.Connection) -> "SourceLineageGraph":
        ensure_source_lineage_schema(db)
        parents: dict[str, set[str]] = {}
        for child_id, child_ver, parent_id, parent_ver in db.execute(
            """
            SELECT child_source_id,child_source_version,parent_source_id,parent_source_version
            FROM source_lineage_edges
            ORDER BY child_source_id,child_source_version,parent_source_id,parent_source_version
            """
        ).fetchall():
            parents.setdefault(source_key(child_id, child_ver), set()).add(source_key(parent_id, parent_ver))
        return cls(parents)

    def roots(self, key: str) -> frozenset[str]:
        cached = self._root_cache.get(key)
        if cached is not None:
            return cached

        def walk(node: str, stack: tuple[str, ...]) -> set[str]:
            if node in stack:
                cycle = " -> ".join(stack + (node,))
                raise ValueError(f"source-lineage cycle detected: {cycle}")
            parents = self.parents.get(node, set())
            if not parents:
                return {node}
            out: set[str] = set()
            for parent in sorted(parents):
                out.update(walk(parent, stack + (node,)))
            return out

        roots = frozenset(walk(key, ()))
        self._root_cache[key] = roots
        return roots

    def components(self, source_keys: Iterable[str]) -> list[LineageComponent]:
        keys = sorted(set(str(x) for x in source_keys))
        if not keys:
            return []
        root_sets = {key: set(self.roots(key)) for key in keys}
        remaining = set(keys)
        out: list[LineageComponent] = []
        while remaining:
            seed = min(remaining)
            component_sources = {seed}
            component_roots = set(root_sets[seed])
            remaining.remove(seed)
            changed = True
            while changed:
                changed = False
                for key in sorted(list(remaining)):
                    if root_sets[key] & component_roots:
                        component_sources.add(key)
                        component_roots.update(root_sets[key])
                        remaining.remove(key)
                        changed = True
            out.append(
                LineageComponent(
                    sources=tuple(sorted(component_sources)),
                    roots=tuple(sorted(component_roots)),
                )
            )
        return sorted(out, key=lambda c: (c.roots, c.sources))

    def independent_count(self, source_keys: Iterable[str]) -> int:
        return len(self.components(source_keys))

    def all_roots(self, source_keys: Iterable[str]) -> set[str]:
        out: set[str] = set()
        for key in source_keys:
            out.update(self.roots(str(key)))
        return out

    @staticmethod
    def novel_component_count(components: Iterable[LineageComponent], seen_roots: set[str]) -> int:
        """Count only components fully disjoint from ancestry already seen.

        A new fine-tune that still descends from any already-seen root is correlated
        and cannot manufacture a new epistemic epoch merely by adding another name.
        """
        return sum(1 for component in components if set(component.roots).isdisjoint(seen_roots))
