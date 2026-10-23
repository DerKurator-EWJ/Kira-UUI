# v0.39 Real Teacher Domain Trial

v0.39 freezes the first larger end-to-end trial in which a computed neural teacher is trained on a fixed domain, Forge is allowed to probe only a subset of that domain, and the teacher is then removed from the Equinox runtime evaluation path.

This benchmark is intentionally **not** a claim about pretrained-world-model distillation. The teacher is a locally trained word-level neural next-token model over a controlled animal domain. The point is to expose extraction/runtime failure modes at a scale larger than the earlier hand-authored fixtures.

Install the neural-trial extra when PyTorch is not already available:

```bash
pip install .[domain-trial]
```

Run:

```bash
eqforge-domain-trial --out out/v0_39_real_teacher_domain_trial
```

The benchmark distinguishes compiled replay, paraphrase transfer, teacher-known but unprobed facts, a completely unprobed relation family, and teacher-known facts outside the compiled domain.
