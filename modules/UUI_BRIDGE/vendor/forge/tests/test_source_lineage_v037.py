from pathlib import Path
import json
import re
import sqlite3
import tempfile

ROOT = Path(__file__).resolve().parents[1]

from equinox_forge.source_lineage import SourceLineageGraph, ensure_source_lineage_schema, register_source_lineage
from equinox_forge.source_lineage_benchmark import run_source_lineage_benchmark

pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
init_text = (ROOT / "equinox_forge" / "__init__.py").read_text(encoding="utf-8")
contract = json.loads((ROOT / "benchmarks" / "v0_37" / "source_lineage_contract.json").read_text(encoding="utf-8"))

project_version = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, flags=re.M).group(1)
init_version = re.search(r'__version__\s*=\s*"([^"]+)"', init_text).group(1)
assert tuple(map(int, project_version.split('.'))) >= (0, 37, 0)
assert init_version == project_version
assert 'eqforge-source-lineage = "equinox_forge.source_lineage_benchmark:main"' in pyproject

# Unit-level ancestry semantics including fallback and cycle rejection.
db = sqlite3.connect(":memory:")
ensure_source_lineage_schema(db)
register_source_lineage(db, "a", "1", "root-x", "1")
register_source_lineage(db, "b", "1", "root-x", "1")
register_source_lineage(db, "blend", "1", "root-x", "1")
register_source_lineage(db, "blend", "1", "root-y", "1")
register_source_lineage(db, "c", "1", "root-y", "1")
graph = SourceLineageGraph.from_db(db)
assert graph.independent_count(["a@1", "b@1"]) == 1
assert graph.independent_count(["a@1", "c@1"]) == 2
assert graph.independent_count(["a@1", "blend@1", "c@1"]) == 1
assert graph.roots("unregistered@1") == frozenset({"unregistered@1"})

cycle_db = sqlite3.connect(":memory:")
ensure_source_lineage_schema(cycle_db)
register_source_lineage(cycle_db, "x", "1", "y", "1")
register_source_lineage(cycle_db, "y", "1", "x", "1")
cycle_graph = SourceLineageGraph.from_db(cycle_db)
try:
    cycle_graph.roots("x@1")
    raise AssertionError("expected lineage cycle rejection")
except ValueError as exc:
    assert "cycle" in str(exc).lower()

with tempfile.TemporaryDirectory() as td:
    report = run_source_lineage_benchmark(Path(td) / "v037")
    h = report.headline_metrics
    assert h["raw_correlated_source_count"] == contract["raw_correlated_sources"]
    assert h["effective_correlated_lineage_count"] == contract["effective_correlated_lineages"]
    assert h["raw_independent_source_count"] == contract["raw_independent_sources"]
    assert h["effective_independent_lineage_count"] == contract["effective_independent_lineages"]
    assert h["bridged_effective_lineage_count"] == contract["multi_parent_bridge_effective_lineages"]
    assert h["lifecycle_status_sequence"] == contract["expected_lifecycle_status_sequence"]
    assert h["stable_runtime_top1"] == contract["stable_runtime_top1"]
    assert h["all_controls_pass"] is True
    assert all(report.controls.values())

print("FORGE_V37_SOURCE_LINEAGE_OK")
