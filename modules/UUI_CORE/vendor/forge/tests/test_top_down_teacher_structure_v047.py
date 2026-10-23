from __future__ import annotations

from pathlib import Path
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from equinox_forge.database import CanonicalDB
from equinox_forge.teacher_behavior import TeacherCandidate
from equinox_forge.top_down_teacher_structure import (
    TOP_DOWN_TEACHER_VERSION,
    TopDownTeacherPolicy,
    extract_top_down_teacher_structure,
    hierarchy_chain,
    run_top_down_teacher_demo,
)

assert TOP_DOWN_TEACHER_VERSION == "0.47"


class MappingTeacher:
    backend_id = "v047-mapping-test-teacher"

    def __init__(self, mapping: dict[str, str]):
        self.mapping = dict(mapping)

    def manifest(self):
        return {"backend_id": self.backend_id, "mapping_size": len(self.mapping)}

    def next_token(self, prompt: str, top_k: int, temperature: float):
        target = self.mapping.get(prompt)
        if target is None:
            return []
        return [TeacherCandidate(1, target, target, 0.99, 5.0)]


with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    report = run_top_down_teacher_demo(td / "demo")
    assert all(report["controls"].values()), report["controls"]
    build = report["teacher_structure_build"]
    assert build["concept_count"] == 9
    assert build["parent_edge_count"] == 8
    assert build["attribute_edge_count"] == 8
    assert build["teacher_queries"] == 16
    assert build["max_hierarchy_depth"] == 4
    assert report["macro_build"]["levels_built"] == 2
    assert report["macro_build"]["macro_count"] == 3

    # The generic extractor is deterministic with respect to seed order.
    mapping = {
        "a is a": "group1",
        "b is a": "group1",
        "c is a": "group2",
        "d is a": "group2",
        "group1 is a": "root",
        "group2 is a": "root",
    }
    teacher = MappingTeacher(mapping)
    policy = TopDownTeacherPolicy(
        parent_prompt_template="{surface} is a",
        stop_surfaces=("root",),
        min_probability=0.5,
        extraction_namespace="v047-order-test",
    )
    results = []
    for i, seeds in enumerate((["a", "b", "c", "d"], ["d", "c", "b", "a"])):
        with CanonicalDB(td / f"order-{i}.sqlite") as cdb:
            bld = extract_top_down_teacher_structure(
                cdb.db, teacher, seed_surfaces=seeds,
                source_id="mapping", source_version="1", policy=policy,
            )
            results.append((bld.digest, bld.top_down_order, hierarchy_chain(cdb.db, bld.run_id, "a")))
    assert results[0] == results[1]
    assert results[0][1][0] == "root"
    assert results[0][2] == ["root", "group1", "a"]

    # Cyclic teacher claims are rejected rather than silently materialized.
    cyc = MappingTeacher({"a is a": "b", "b is a": "a"})
    with CanonicalDB(td / "cycle.sqlite") as cdb:
        try:
            extract_top_down_teacher_structure(
                cdb.db, cyc, seed_surfaces=["a"], source_id="cycle", source_version="1",
                policy=TopDownTeacherPolicy(
                    parent_prompt_template="{surface} is a",
                    min_probability=0.5,
                    max_parent_depth=4,
                    extraction_namespace="v047-cycle",
                ),
            )
        except ValueError as exc:
            assert "cycle" in str(exc).lower()
        else:
            raise AssertionError("cyclic teacher hierarchy must be rejected")

print("FORGE_V047_TOP_DOWN_TEACHER_STRUCTURE_OK")
