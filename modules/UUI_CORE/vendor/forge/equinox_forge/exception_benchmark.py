from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable
import json

from .runtime_cognition import RuntimeCognition, RuntimeCognitionPolicy
from .runtime_pack import LoadedRuntimePack, load_base_pack

EXCEPTION_BENCHMARK_FORMAT = "equinox-forge-exception-benchmark-v1"
EXCEPTION_BENCHMARK_VERSION = "0.28"


@dataclass(frozen=True)
class ExceptionCase:
    context: str
    expected: str | None = None
    note: str | None = None


@dataclass
class ExceptionBenchmarkReport:
    format: str
    version: str
    cases: int
    baseline_top1_accuracy: float | None
    exception_top1_accuracy: float | None
    changed_top1_cases: int
    improved_cases: int
    regressed_cases: int
    exception_frames_used: int
    details: list[dict[str, Any]]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_exception_cases(paths: Iterable[str | Path]) -> list[ExceptionCase]:
    out: list[ExceptionCase] = []
    for raw in paths:
        path = Path(raw)
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if path.suffix.lower() == ".jsonl":
                obj = json.loads(line)
                if not isinstance(obj, dict) or not obj.get("context"):
                    raise ValueError(f"{path}:{line_no}: expected JSON object with context")
                out.append(ExceptionCase(
                    str(obj["context"]),
                    str(obj["expected"]) if obj.get("expected") is not None else None,
                    str(obj["note"]) if obj.get("note") is not None else None,
                ))
            else:
                parts = line.split("\t")
                if not parts[0].strip():
                    raise ValueError(f"{path}:{line_no}: empty context")
                out.append(ExceptionCase(
                    parts[0].strip(),
                    parts[1].strip() if len(parts) > 1 and parts[1].strip() else None,
                    parts[2].strip() if len(parts) > 2 and parts[2].strip() else None,
                ))
    return out


def _top1(report) -> str | None:
    return report.predictions[0].surface if report.predictions else None


def benchmark_exception_overrides(
    pack: LoadedRuntimePack | str | Path,
    cases: list[ExceptionCase],
    *,
    policy: RuntimeCognitionPolicy | None = None,
) -> ExceptionBenchmarkReport:
    loaded = load_base_pack(pack) if isinstance(pack, (str, Path)) else pack
    base = policy or RuntimeCognitionPolicy()
    baseline_policy = replace(base, enable_exception_override_frames=False, enable_frame_specificity_override=True)
    exception_policy = replace(base, enable_exception_override_frames=True, enable_frame_specificity_override=True)
    baseline = RuntimeCognition(loaded, baseline_policy)
    enabled = RuntimeCognition(loaded, exception_policy)

    labelled = base_correct = exc_correct = changed = improved = regressed = used = 0
    details: list[dict[str, Any]] = []
    for case in cases:
        old = baseline.predict(case.context)
        new = enabled.predict(case.context)
        old_top = _top1(old); new_top = _top1(new)
        if old_top != new_top:
            changed += 1
        old_ok = new_ok = None
        if case.expected is not None:
            labelled += 1
            old_ok = old_top == case.expected
            new_ok = new_top == case.expected
            base_correct += int(old_ok)
            exc_correct += int(new_ok)
            improved += int((not old_ok) and new_ok)
            regressed += int(old_ok and (not new_ok))
        admitted_exception_ids = [
            str(x.get("frame_node_id")) for x in new.frame_admission
            if x.get("admitted") and not x.get("suppressed_by_specificity")
            and _frame_is_exception(loaded, str(x.get("frame_node_id")))
        ]
        used += int(bool(admitted_exception_ids))
        details.append({
            "context": case.context,
            "expected": case.expected,
            "note": case.note,
            "baseline_top1": old_top,
            "exception_top1": new_top,
            "baseline_correct": old_ok,
            "exception_correct": new_ok,
            "used_exception_frame_ids": admitted_exception_ids,
            "baseline_resolved_context_ids": list(old.resolved_context_ids),
            "exception_resolved_context_ids": list(new.resolved_context_ids),
            "exception_frame_admission": list(new.frame_admission),
        })

    return ExceptionBenchmarkReport(
        format=EXCEPTION_BENCHMARK_FORMAT,
        version=EXCEPTION_BENCHMARK_VERSION,
        cases=len(cases),
        baseline_top1_accuracy=(base_correct / labelled if labelled else None),
        exception_top1_accuracy=(exc_correct / labelled if labelled else None),
        changed_top1_cases=changed,
        improved_cases=improved,
        regressed_cases=regressed,
        exception_frames_used=used,
        details=details,
        notes=[
            "The benchmark executes the same immutable BASE twice. Only automatic exception frames are disabled/enabled; no recompilation occurs between runs.",
            "v0.27 specificity resolution remains enabled in both runs so a subtype override can categorically suppress a broader inherited rule.",
            "A benchmark improvement therefore measures utility of the compiled exception hypothesis, not additional teacher information injected at evaluation time.",
        ],
    )


def _frame_is_exception(pack: LoadedRuntimePack, node_id: str) -> bool:
    for node in pack.nodes:
        if node.node_id == node_id:
            return bool((node.metadata or {}).get("exception_override"))
    return False
