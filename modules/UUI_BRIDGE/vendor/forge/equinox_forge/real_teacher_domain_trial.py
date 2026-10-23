from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import argparse
import json
import random
import time

from .behaviour_retention import (
    benchmark_teacher_probes_against_pack,
    compile_and_benchmark_teacher_behaviour,
)
from .runtime_cognition import RuntimeCognitionPolicy
from .teacher_behavior import BehaviourProbeConfig, TeacherCandidate, extract_teacher_behaviour

TRIAL_FORMAT = "equinox-forge-real-teacher-domain-trial-v1"
TRIAL_VERSION = "0.39"

# A fixed, intentionally modest domain. The goal is not zoological completeness;
# it is a deterministic knowledge space large enough to expose extraction/runtime
# failure modes while keeping the neural teacher locally reproducible.
ANIMALS: list[dict[str, str]] = [
    {"entity":"hund","class":"säugetier","habitat":"land","ability":"laufen","trait":"fell","food":"fleisch"},
    {"entity":"katze","class":"säugetier","habitat":"land","ability":"klettern","trait":"fell","food":"fleisch"},
    {"entity":"wolf","class":"säugetier","habitat":"wald","ability":"laufen","trait":"fell","food":"fleisch"},
    {"entity":"fuchs","class":"säugetier","habitat":"wald","ability":"graben","trait":"fell","food":"fleisch"},
    {"entity":"pferd","class":"säugetier","habitat":"land","ability":"galoppieren","trait":"fell","food":"gras"},
    {"entity":"kuh","class":"säugetier","habitat":"land","ability":"laufen","trait":"fell","food":"gras"},
    {"entity":"schaf","class":"säugetier","habitat":"land","ability":"laufen","trait":"wolle","food":"gras"},
    {"entity":"ziege","class":"säugetier","habitat":"berg","ability":"klettern","trait":"fell","food":"gras"},
    {"entity":"löwe","class":"säugetier","habitat":"savanne","ability":"jagen","trait":"fell","food":"fleisch"},
    {"entity":"elefant","class":"säugetier","habitat":"savanne","ability":"laufen","trait":"rüssel","food":"pflanzen"},
    {"entity":"affe","class":"säugetier","habitat":"wald","ability":"klettern","trait":"fell","food":"früchte"},
    {"entity":"bär","class":"säugetier","habitat":"wald","ability":"klettern","trait":"fell","food":"fische"},
    {"entity":"adler","class":"vogel","habitat":"berg","ability":"fliegen","trait":"federn","food":"fleisch"},
    {"entity":"falke","class":"vogel","habitat":"himmel","ability":"fliegen","trait":"federn","food":"fleisch"},
    {"entity":"eule","class":"vogel","habitat":"wald","ability":"fliegen","trait":"federn","food":"fleisch"},
    {"entity":"spatz","class":"vogel","habitat":"himmel","ability":"fliegen","trait":"federn","food":"körner"},
    {"entity":"ente","class":"vogel","habitat":"wasser","ability":"schwimmen","trait":"federn","food":"pflanzen"},
    {"entity":"pinguin","class":"vogel","habitat":"eis","ability":"schwimmen","trait":"federn","food":"fische"},
    {"entity":"hai","class":"fisch","habitat":"meer","ability":"schwimmen","trait":"flossen","food":"fische"},
    {"entity":"lachs","class":"fisch","habitat":"wasser","ability":"schwimmen","trait":"schuppen","food":"insekten"},
    {"entity":"karpfen","class":"fisch","habitat":"wasser","ability":"schwimmen","trait":"schuppen","food":"pflanzen"},
    {"entity":"forelle","class":"fisch","habitat":"wasser","ability":"schwimmen","trait":"schuppen","food":"insekten"},
    {"entity":"krokodil","class":"reptil","habitat":"wasser","ability":"schwimmen","trait":"schuppen","food":"fleisch"},
    {"entity":"schlange","class":"reptil","habitat":"land","ability":"kriechen","trait":"schuppen","food":"fleisch"},
    {"entity":"eidechse","class":"reptil","habitat":"land","ability":"kriechen","trait":"schuppen","food":"insekten"},
    {"entity":"schildkröte","class":"reptil","habitat":"land","ability":"kriechen","trait":"panzer","food":"pflanzen"},
    {"entity":"frosch","class":"amphibie","habitat":"wasser","ability":"springen","trait":"haut","food":"insekten"},
    {"entity":"kröte","class":"amphibie","habitat":"land","ability":"springen","trait":"haut","food":"insekten"},
    {"entity":"biene","class":"insekt","habitat":"wiese","ability":"fliegen","trait":"flügel","food":"nektar"},
    {"entity":"ameise","class":"insekt","habitat":"land","ability":"laufen","trait":"panzer","food":"pflanzen"},
    {"entity":"käfer","class":"insekt","habitat":"wald","ability":"fliegen","trait":"panzer","food":"pflanzen"},
    {"entity":"libelle","class":"insekt","habitat":"wasser","ability":"fliegen","trait":"flügel","food":"insekten"},
]

CONTROLS: list[dict[str, str]] = [
    {"entity":"stein","class":"objekt"}, {"entity":"tisch","class":"objekt"},
    {"entity":"lampe","class":"objekt"}, {"entity":"buch","class":"objekt"},
    {"entity":"hammer","class":"objekt"}, {"entity":"glas","class":"objekt"},
    {"entity":"stuhl","class":"objekt"}, {"entity":"uhr","class":"objekt"},
]

RELATIONS: tuple[tuple[str, str], ...] = (
    ("class", "{entity} ist ein"),
    ("habitat", "{entity} lebt im"),
    ("ability", "{entity} kann"),
    ("trait", "{entity} hat"),
)
OPAQUE_RELATION = ("food", "{entity} frisst")


def _article_variant(prompt: str) -> str:
    return "der " + prompt


def _fact_prompts(row: dict[str, str]) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for key, template in RELATIONS:
        p = template.format(entity=row["entity"])
        out.append((key, p, row[key]))
    key, template = OPAQUE_RELATION
    p = template.format(entity=row["entity"])
    out.append((key, p, row[key]))
    return out


class _NeuralDomainTeacher:
    backend_id = "v039-local-neural-domain-teacher"

    def __init__(self, examples: list[tuple[str, str]], out_dir: Path, *, seed: int = 3901, max_epochs: int = 500):
        try:
            import torch
            import torch.nn as nn
            from torch.nn.utils.rnn import pack_padded_sequence
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("v0.39 domain trial requires PyTorch") from exc
        self.torch = torch
        self.nn = nn
        self.pack_padded_sequence = pack_padded_sequence
        random.seed(seed)
        torch.manual_seed(seed)
        tokens = {"<pad>"}
        for prompt, target in examples:
            tokens.update(prompt.split())
            tokens.add(target)
        self.vocab = ["<pad>"] + sorted(x for x in tokens if x != "<pad>")
        self.idx = {w: i for i, w in enumerate(self.vocab)}
        self.pad = self.idx["<pad>"]
        self.seed = seed
        self.model = self._build_model(len(self.vocab))
        self.training_examples = len(examples)
        self.training_elapsed_ms = 0.0
        self.training_epochs = 0
        self.training_accuracy = 0.0
        self._train(examples, max_epochs=max_epochs)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = out_dir / "teacher-state.pt"
        torch.save({
            "state_dict": self.model.state_dict(),
            "vocab": self.vocab,
            "seed": seed,
            "training_examples": len(examples),
        }, self.model_path)

    def _build_model(self, vocab_size: int):
        nn = self.nn
        pack_padded_sequence = self.pack_padded_sequence
        pad = self.pad
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(vocab_size, 28, padding_idx=pad)
                self.gru = nn.GRU(28, 56, batch_first=True)
                self.out = nn.Linear(56, vocab_size)
            def forward(self, x, lengths):
                emb = self.emb(x)
                packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
                _, h = self.gru(packed)
                return self.out(h[-1])
        return Model()

    def _encode(self, prompts: list[str]):
        torch = self.torch
        ids = [[self.idx[w] for w in p.split()] for p in prompts]
        lengths = torch.tensor([len(x) for x in ids], dtype=torch.long)
        width = int(lengths.max().item())
        x = torch.full((len(ids), width), self.pad, dtype=torch.long)
        for i, seq in enumerate(ids):
            x[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        return x, lengths

    def _train(self, examples: list[tuple[str, str]], *, max_epochs: int):
        torch = self.torch
        prompts = [p for p, _ in examples]
        targets = torch.tensor([self.idx[t] for _, t in examples], dtype=torch.long)
        x, lengths = self._encode(prompts)
        opt = torch.optim.AdamW(self.model.parameters(), lr=0.025, weight_decay=0.0005)
        loss_fn = self.nn.CrossEntropyLoss()
        start = time.perf_counter()
        stable = 0
        for epoch in range(1, max_epochs + 1):
            self.model.train()
            logits = self.model(x, lengths)
            loss = loss_fn(logits, targets)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                acc = float((logits.argmax(dim=-1) == targets).float().mean().item())
            if acc >= 0.999:
                stable += 1
            else:
                stable = 0
            if stable >= 8:
                self.training_epochs = epoch
                break
        else:
            self.training_epochs = max_epochs
        self.training_elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x, lengths)
            self.training_accuracy = float((logits.argmax(dim=-1) == targets).float().mean().item())
        if self.training_accuracy < 0.995:
            raise RuntimeError(f"teacher failed to fit controlled domain: accuracy={self.training_accuracy:.4f}")

    def manifest(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "model_dir": str(self.model_path),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "vocab_size": len(self.vocab),
            "training_examples": self.training_examples,
            "training_epochs": self.training_epochs,
            "training_accuracy": self.training_accuracy,
            "training_elapsed_ms": self.training_elapsed_ms,
            "seed": self.seed,
            "architecture": "Embedding(28)-GRU(56)-Linear",
        }

    def next_token(self, prompt: str, top_k: int, temperature: float) -> list[TeacherCandidate]:
        torch = self.torch
        words = prompt.split()
        if any(w not in self.idx for w in words):
            return []
        x, lengths = self._encode([prompt])
        with torch.no_grad():
            logits = self.model(x, lengths)[0].float()
            probs = torch.softmax(logits / temperature, dim=-1)
            vals, idxs = torch.topk(probs, min(top_k, len(self.vocab)))
        out: list[TeacherCandidate] = []
        for prob, idx in zip(vals, idxs):
            i = int(idx)
            token = self.vocab[i]
            if token == "<pad>":
                continue
            out.append(TeacherCandidate(i, token, token, float(prob), float(logits[i])))
        return out


@dataclass
class TrialCase:
    category: str
    prompt: str
    expected: str
    teacher_top1: str | None
    runtime_top1: str | None
    teacher_correct: bool
    runtime_correct: bool
    runtime_abstained: bool
    runtime_matches_teacher: bool
    recall_nodes: int
    relation_checks: int

    def to_dict(self) -> dict[str, Any]: return asdict(self)


def _summary(rows: list[TrialCase]) -> dict[str, Any]:
    if not rows:
        return {"cases": 0}
    n = len(rows)
    return {
        "cases": n,
        "teacher_accuracy": sum(r.teacher_correct for r in rows) / n,
        "runtime_accuracy": sum(r.runtime_correct for r in rows) / n,
        "runtime_teacher_top1_retention": sum(r.runtime_matches_teacher for r in rows) / n,
        "runtime_abstention_rate": sum(r.runtime_abstained for r in rows) / n,
        "average_recall_nodes": sum(r.recall_nodes for r in rows) / n,
        "average_relation_checks": sum(r.relation_checks for r in rows) / n,
    }


def _training_examples(animals: list[dict[str, str]], controls: list[dict[str, str]]) -> list[tuple[str, str]]:
    examples: list[tuple[str, str]] = []
    for row in animals:
        for _, prompt, target in _fact_prompts(row):
            examples.append((prompt, target))
            examples.append((_article_variant(prompt), target))
    for row in controls:
        p = f"{row['entity']} ist ein"
        examples.append((p, row["class"]))
        examples.append((_article_variant(p), row["class"]))
    return examples


def _compile_split(animals: list[dict[str, str]]) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    compiled: list[tuple[str, str]] = []
    heldout: list[tuple[str, str]] = []
    opaque: list[tuple[str, str]] = []
    for i, row in enumerate(animals):
        facts = _fact_prompts(row)
        # Rotate the 2-of-4 core relation split so every relation is represented in
        # compilation while every entity retains two genuinely unprobed facts.
        keep = {i % 4, (i + 1) % 4}
        for j, (_, prompt, target) in enumerate(facts[:4]):
            (compiled if j in keep else heldout).append((prompt, target))
        _, prompt, target = facts[4]
        opaque.append((prompt, target))
    return compiled, heldout, opaque


def run_domain_trial(out_dir: str | Path, *, max_entities: int | None = None, seed: int = 3901) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    animals = ANIMALS[:max_entities] if max_entities else list(ANIMALS)
    if len(animals) < 8:
        raise ValueError("domain trial requires at least 8 entities")
    training = _training_examples(animals, CONTROLS)
    teacher = _NeuralDomainTeacher(training, out, seed=seed)
    compiled, heldout, opaque = _compile_split(animals)

    probe_cfg = BehaviourProbeConfig(
        top_k=4,
        min_probability=0.00001,
        evidence_confidence=0.95,
        include_punctuation_only=True,
    )
    policy = RuntimeCognitionPolicy(
        max_recall_nodes=96,
        max_recall_depth=3,
        max_predictions=4,
        min_prediction_score=0.000001,
        # v0.39 is a frozen measurement baseline. Later runtime releases must not
        # silently rewrite its reference numbers.
        enable_exact_context_authority=False,
    )
    compile_dir = out / "compiled"
    replay_report = compile_and_benchmark_teacher_behaviour(
        teacher,
        [p for p, _ in compiled],
        compile_dir,
        "v039-neural-domain-teacher",
        "1",
        probe_cfg=probe_cfg,
        cognition_policy=policy,
        runtime_top_k=4,
        pack_id="equinox-v039-domain",
        pack_version=39,
    )
    pack_path = compile_dir / "retention.eqbase"

    categories: dict[str, list[tuple[str, str]]] = {
        "replay": compiled,
        "paraphrase": [(_article_variant(p), target) for p, target in compiled],
        "heldout_fact": heldout,
        "opaque_relation": opaque,
        "outside_domain_control": [(f"{row['entity']} ist ein", row["class"]) for row in CONTROLS],
    }
    rows: list[TrialCase] = []
    raw_eval: dict[str, Any] = {}
    for category, cases in categories.items():
        _, _, probes = extract_teacher_behaviour(
            teacher,
            [p for p, _ in cases],
            f"v039-eval-{category}",
            "1",
            probe_cfg,
        )
        bench_rows, elapsed = benchmark_teacher_probes_against_pack(
            pack_path,
            probes,
            probe_cfg,
            cognition_policy=policy,
            teacher_top_k=4,
            runtime_top_k=4,
        )
        raw_eval[category] = {"runtime_eval_elapsed_ms": elapsed, "cases": [x.to_dict() for x in bench_rows]}
        for (prompt, expected), b in zip(cases, bench_rows):
            rows.append(TrialCase(
                category=category,
                prompt=prompt,
                expected=expected,
                teacher_top1=b.teacher_top1,
                runtime_top1=b.runtime_top1,
                teacher_correct=(b.teacher_top1 == expected),
                runtime_correct=(b.runtime_top1 == expected),
                runtime_abstained=(b.runtime_top1 is None),
                runtime_matches_teacher=(b.teacher_top1 is not None and b.teacher_top1 == b.runtime_top1),
                recall_nodes=b.recall_nodes,
                relation_checks=b.relation_checks,
            ))

    by_category = {c: _summary([r for r in rows if r.category == c]) for c in categories}
    for category in categories:
        elapsed = float(raw_eval[category]["runtime_eval_elapsed_ms"])
        count = int(by_category[category].get("cases", 0))
        by_category[category]["runtime_eval_elapsed_ms"] = elapsed
        by_category[category]["runtime_eval_ms_per_case"] = elapsed / count if count else 0.0
    retained_categories = ["replay", "paraphrase", "heldout_fact", "opaque_relation"]
    retained_rows = [r for r in rows if r.category in retained_categories]
    controls = [r for r in rows if r.category == "outside_domain_control"]
    teacher_manifest = teacher.manifest()
    pack_bytes = pack_path.stat().st_size
    teacher_bytes = teacher.model_path.stat().st_size
    report = {
        "format": TRIAL_FORMAT,
        "version": TRIAL_VERSION,
        "scope": "locally-trained-neural-teacher-controlled-animal-domain",
        "general_world_model_claim": False,
        "pretrained_external_model_claim": False,
        "teacher_is_computed_neural_model": True,
        "domain_entities": len(animals),
        "domain_core_relations": len(RELATIONS),
        "opaque_relation_families": 1,
        "teacher_training_examples": len(training),
        "compiled_probe_facts": len(compiled),
        "teacher_queries_for_compilation": len(compiled),
        "teacher_query_fraction_of_animal_fact_space": len(compiled) / (len(animals) * 5),
        "uncompiled_core_facts": len(heldout),
        "opaque_relation_facts": len(opaque),
        "teacher": teacher_manifest,
        "pack_bytes": pack_bytes,
        "teacher_state_bytes": teacher_bytes,
        "pack_to_teacher_state_ratio": pack_bytes / teacher_bytes if teacher_bytes else None,
        "pack_nodes": replay_report.pack_nodes,
        "pack_relations": replay_report.pack_relations,
        "compiled_prediction_evidence": replay_report.compiled_prediction_evidence,
        "replay_compiler_report": replay_report.to_dict(),
        "categories": by_category,
        "retention_over_in_domain_categories": _summary(retained_rows),
        "outside_domain_abstention_rate": (sum(r.runtime_abstained for r in controls) / len(controls) if controls else 0.0),
        "cases": [r.to_dict() for r in rows],
        "raw_evaluation": raw_eval,
        "notes": [
            "The teacher is a real locally trained neural next-token model, not a scripted lookup backend.",
            "The domain is fixed and controlled; this is not a claim about general-world LLM distillation.",
            "The neural teacher is trained on the full domain, but Forge is allowed to probe only the compiled subset.",
            "heldout_fact measures facts known by the teacher but never directly probed into Equinox.",
            "opaque_relation measures an entire relation family known by the teacher but never probed into Equinox.",
            "outside_domain_control measures whether Equinox abstains on teacher-known facts outside the compiled animal domain.",
        ],
    }
    (out / "domain-trial-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Equinox Forge v0.39 real neural teacher domain trial")
    ap.add_argument("--out", default="out/v0_39_real_teacher_domain_trial")
    ap.add_argument("--entities", type=int, default=None, help="optional prefix size for a faster smoke trial (min 8)")
    ap.add_argument("--seed", type=int, default=3901)
    args = ap.parse_args(argv)
    report = run_domain_trial(args.out, max_entities=args.entities, seed=args.seed)
    print(json.dumps({
        "format": report["format"],
        "version": report["version"],
        "domain_entities": report["domain_entities"],
        "teacher_training_accuracy": report["teacher"]["training_accuracy"],
        "pack_bytes": report["pack_bytes"],
        "teacher_state_bytes": report["teacher_state_bytes"],
        "pack_to_teacher_state_ratio": report["pack_to_teacher_state_ratio"],
        "categories": report["categories"],
        "outside_domain_abstention_rate": report["outside_domain_abstention_rate"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
