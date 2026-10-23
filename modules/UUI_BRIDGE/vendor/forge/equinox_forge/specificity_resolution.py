from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable
import json

from .runtime_cognition import RuntimeCognition, RuntimeCognitionPolicy
from .runtime_pack import LoadedRuntimePack, load_base_pack

SPECIFICITY_RESOLUTION_FORMAT = "equinox-forge-specificity-resolution-v1"
SPECIFICITY_RESOLUTION_VERSION = "0.27"


@dataclass(frozen=True)
class SpecificityCase:
    context: str
    expected: str | None = None
    note: str | None = None


@dataclass
class SpecificityResolutionReport:
    format: str
    version: str
    cases: int
    baseline_top1_accuracy: float | None
    specificity_top1_accuracy: float | None
    changed_top1_cases: int
    improved_cases: int
    regressed_cases: int
    specificity_suppressed_frames: int
    details: list[dict[str, Any]]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_specificity_cases(paths: Iterable[str | Path]) -> list[SpecificityCase]:
    out: list[SpecificityCase] = []
    for raw in paths:
        path = Path(raw)
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if path.suffix.lower() == ".jsonl":
                obj = json.loads(line)
                if not isinstance(obj, dict) or not obj.get("context"):
                    raise ValueError(f"{path}:{line_no}: expected JSON object with context")
                out.append(SpecificityCase(str(obj["context"]),
                                           str(obj["expected"]) if obj.get("expected") is not None else None,
                                           str(obj["note"]) if obj.get("note") is not None else None))
            else:
                parts = line.split("\t")
                if not parts[0].strip():
                    raise ValueError(f"{path}:{line_no}: empty context")
                out.append(SpecificityCase(parts[0].strip(), parts[1].strip() if len(parts) > 1 and parts[1].strip() else None,
                                           parts[2].strip() if len(parts) > 2 and parts[2].strip() else None))
    return out


def _top1(report) -> str | None:
    return report.predictions[0].surface if report.predictions else None


def benchmark_specificity_resolution(
    pack: LoadedRuntimePack | str | Path,
    cases: list[SpecificityCase],
    *,
    policy: RuntimeCognitionPolicy | None = None,
) -> SpecificityResolutionReport:
    loaded = load_base_pack(pack) if isinstance(pack, (str, Path)) else pack
    base = policy or RuntimeCognitionPolicy()
    baseline_policy = replace(base, enable_frame_specificity_override=False)
    specificity_policy = replace(base, enable_frame_specificity_override=True)
    baseline = RuntimeCognition(loaded, baseline_policy)
    specific = RuntimeCognition(loaded, specificity_policy)

    baseline_correct = specific_correct = labelled = changed = improved = regressed = suppressed = 0
    details: list[dict[str, Any]] = []
    for case in cases:
        old = baseline.predict(case.context)
        new = specific.predict(case.context)
        old_top = _top1(old); new_top = _top1(new)
        if old_top != new_top:
            changed += 1
        old_ok = new_ok = None
        if case.expected is not None:
            labelled += 1
            old_ok = old_top == case.expected
            new_ok = new_top == case.expected
            baseline_correct += int(old_ok)
            specific_correct += int(new_ok)
            improved += int((not old_ok) and new_ok)
            regressed += int(old_ok and (not new_ok))
        case_suppressed = int(new.work.get("frame_specificity_suppressed", 0))
        suppressed += case_suppressed
        details.append({
            "context": case.context,
            "expected": case.expected,
            "note": case.note,
            "baseline_top1": old_top,
            "specificity_top1": new_top,
            "baseline_correct": old_ok,
            "specificity_correct": new_ok,
            "specificity_suppressed_frames": case_suppressed,
            "specificity_resolved_context_ids": list(new.resolved_context_ids),
            "baseline_resolved_context_ids": list(old.resolved_context_ids),
            "specificity_frame_admission": list(new.frame_admission),
        })

    return SpecificityResolutionReport(
        format=SPECIFICITY_RESOLUTION_FORMAT,
        version=SPECIFICITY_RESOLUTION_VERSION,
        cases=len(cases),
        baseline_top1_accuracy=(baseline_correct / labelled if labelled else None),
        specificity_top1_accuracy=(specific_correct / labelled if labelled else None),
        changed_top1_cases=changed,
        improved_cases=improved,
        regressed_cases=regressed,
        specificity_suppressed_frames=suppressed,
        details=details,
        notes=[
            "Exact predictive contexts are resolved before frame schemas and therefore remain the highest-specificity runtime path.",
            "Within equal-width frame schemas, direct prototype/typed scopes outrank hierarchical transferred scopes; suppressed broader frames contribute no prediction score.",
            "The benchmark compares the same immutable BASE with specificity resolution disabled versus enabled; no recompilation occurs between runs.",
        ],
    )
