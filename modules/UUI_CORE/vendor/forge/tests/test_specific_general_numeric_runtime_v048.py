from __future__ import annotations

from pathlib import Path
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from equinox_forge.numeric_runtime import (
    NUMERIC_RUNTIME_FORMAT,
    NUMERIC_RUNTIME_VERSION,
    NumericRuntimeFact,
    NumericRuntimePack,
    NumericRuntimePolicy,
    SpecificToGeneralNumericRuntime,
    run_numeric_runtime_demo,
)

assert NUMERIC_RUNTIME_VERSION == "0.48"

with tempfile.TemporaryDirectory() as td:
    report = run_numeric_runtime_demo(Path(td) / "demo")
    assert all(report["controls"].values()), report["controls"]
    q = report["queries"]
    assert q["owl_alive"]["inheritance_depth"] == 2
    assert q["owl_covering"]["inheritance_depth"] == 1
    assert q["penguin_can_fly"]["inheritance_depth"] == 0
    assert q["weakbird_can_fly"]["inheritance_depth"] == 1

# Same-depth disagreement must abstain instead of selecting a parent by order.
# IDs are intentionally opaque; only relation structure matters.
IS_A, STATE = 10, 20
child, left, right, yes, no = 100, 101, 102, 201, 202
ambiguous_pack = NumericRuntimePack(
    NUMERIC_RUNTIME_FORMAT, NUMERIC_RUNTIME_VERSION, 1, "x", (IS_A,),
    [
        NumericRuntimeFact(child, IS_A, left, 1.0, 1.0, 1, 1),
        NumericRuntimeFact(child, IS_A, right, 1.0, 1.0, 1, 1),
        NumericRuntimeFact(left, STATE, yes, 0.9, 0.9, 1, 1),
        NumericRuntimeFact(right, STATE, no, 0.9, 0.9, 1, 1),
    ],
    [], "manual", {},
)
rt = SpecificToGeneralNumericRuntime(
    ambiguous_pack, NumericRuntimePolicy((IS_A,), min_quality=0.5, ambiguity_margin=0.05)
)
res = rt.resolve(child, STATE)
assert res.status == "ambiguous", res
assert set(res.candidate_object_ids) == {yes, no}
assert res.inheritance_depth == 1

# A cycle in parent edges must terminate deterministically rather than recurse forever.
cycle_pack = NumericRuntimePack(
    NUMERIC_RUNTIME_FORMAT, NUMERIC_RUNTIME_VERSION, 1, "x", (IS_A,),
    [
        NumericRuntimeFact(1, IS_A, 2, 1.0, 1.0, 1, 1),
        NumericRuntimeFact(2, IS_A, 1, 1.0, 1.0, 1, 1),
        NumericRuntimeFact(2, STATE, yes, 0.9, 0.9, 1, 1),
    ],
    [], "manual", {},
)
cycle_rt = SpecificToGeneralNumericRuntime(cycle_pack, NumericRuntimePolicy((IS_A,), max_parent_depth=8))
cycle_result = cycle_rt.resolve(1, STATE)
assert cycle_result.status == "resolved"
assert cycle_result.object_id == yes
assert cycle_result.inheritance_depth == 1
assert cycle_result.inheritance_path == (1, 2)

print("FORGE_V048_SPECIFIC_GENERAL_NUMERIC_RUNTIME_OK")
