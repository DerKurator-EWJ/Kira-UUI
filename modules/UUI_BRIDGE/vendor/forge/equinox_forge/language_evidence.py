from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
import math
import re
from typing import Iterable

from .canonical import make_text_surface, make_composite, normalize_surface
from .ir import ForgeDocument, IRNode, RelationEvidence, RelationType, SourceRef, NodeKind, Modality

# Unicode-aware enough for the intended first controlled German/European corpus pass.
# Words may contain apostrophes/hyphens internally; punctuation is tokenized separately.
_TOKEN_RE = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*|[^\w\s]", re.UNICODE)
_SENTENCE_END = {".", "!", "?"}


@dataclass
class LanguageEvidenceConfig:
    lowercase: bool = False
    include_punctuation: bool = False
    max_context: int = 3
    min_transition_count: int = 1
    min_prediction_count: int = 1
    min_phrase_count: int = 2
    max_phrase_words: int = 4
    max_substitutions_per_word: int = 8
    min_substitution_score: float = 0.10
    evidence_confidence: float = 0.80


@dataclass
class LanguageEvidenceReport:
    documents: int = 0
    sentences: int = 0
    lexical_tokens: int = 0
    unique_surfaces: int = 0
    q_nodes: int = 0
    phrase_nodes: int = 0
    transition_evidence: int = 0
    prediction_evidence: int = 0
    substitution_evidence: int = 0

    def to_dict(self):
        return asdict(self)


def tokenize_text(text: str, cfg: LanguageEvidenceConfig) -> list[list[str]]:
    """Return sentence-like token sequences.

    Sentence boundaries are structural and are not emitted as lexical tokens unless
    include_punctuation is enabled. This prevents transitions from leaking across
    sentence boundaries while keeping the canonical text Q alphabet clean.
    """
    raw = _TOKEN_RE.findall(normalize_surface(text))
    sentences: list[list[str]] = []
    current: list[str] = []
    for tok in raw:
        if cfg.lowercase and tok.isalpha():
            tok = tok.lower()
        is_word = bool(re.match(r"^[^\W_]+", tok, re.UNICODE))
        if is_word or cfg.include_punctuation:
            current.append(tok)
        if tok in _SENTENCE_END:
            if current:
                # If punctuation is disabled it wasn't appended, but the preceding words remain.
                sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return [s for s in sentences if s]


def _dedupe_nodes(nodes: Iterable[IRNode]) -> list[IRNode]:
    return list({n.node_id: n for n in nodes}.values())


def _weighted_jaccard(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    num = sum(min(a[k], b[k]) for k in keys)
    den = sum(max(a[k], b[k]) for k in keys)
    return (num / den) if den else 0.0


def build_controlled_language_evidence(
    texts: list[str],
    source_id: str,
    source_version: str = "local",
    cfg: LanguageEvidenceConfig | None = None,
) -> tuple[ForgeDocument, LanguageEvidenceReport]:
    cfg = cfg or LanguageEvidenceConfig()
    if cfg.max_context < 1:
        raise ValueError("max_context must be >= 1")
    if cfg.max_phrase_words < 2:
        raise ValueError("max_phrase_words must be >= 2")

    sentences: list[list[str]] = []
    for text in texts:
        sentences.extend(tokenize_text(text, cfg))

    surface_nodes: dict[str, IRNode] = {}
    all_nodes: dict[str, IRNode] = {}
    for sent in sentences:
        for token in sent:
            if token in surface_nodes:
                continue
            qs, node = make_text_surface(token)
            for q in qs:
                all_nodes[q.node_id] = q
            surface_nodes[token] = node
            all_nodes[node.node_id] = node

    source = SourceRef(source_id=source_id, source_version=source_version, extraction="controlled_corpus_v1")
    evidence: list[RelationEvidence] = []

    # Directed bigram transitions: P(next | current).
    transition_counts: dict[str, Counter] = defaultdict(Counter)
    for sent in sentences:
        for a, b in zip(sent, sent[1:]):
            transition_counts[a][b] += 1
    for a, nexts in transition_counts.items():
        total = sum(nexts.values())
        for b, count in sorted(nexts.items()):
            if count < cfg.min_transition_count:
                continue
            evidence.append(RelationEvidence(
                surface_nodes[a].node_id, surface_nodes[b].node_id,
                RelationType.TRANSITION.value, count / total,
                cfg.evidence_confidence, source,
                {"count": count, "context_total": total, "estimator": "mle_bigram"},
            ))

    # Context -> next-token prediction. Context is a canonical ordered composite,
    # so prediction is directed and independent from a specific teacher tokenizer.
    prediction_counts: dict[tuple[str, ...], Counter] = defaultdict(Counter)
    for sent in sentences:
        for i in range(1, len(sent)):
            for width in range(1, min(cfg.max_context, i) + 1):
                ctx = tuple(sent[i-width:i])
                prediction_counts[ctx][sent[i]] += 1
    context_nodes: dict[tuple[str, ...], IRNode] = {}
    for ctx, targets in prediction_counts.items():
        members = [surface_nodes[t] for t in ctx]
        label = " ".join(ctx)
        cnode = make_composite(Modality.TEXT.value, members, NodeKind.TEMPLATE.value, label=label)
        context_nodes[ctx] = cnode
        all_nodes[cnode.node_id] = cnode
        total = sum(targets.values())
        for target, count in sorted(targets.items()):
            if count < cfg.min_prediction_count:
                continue
            evidence.append(RelationEvidence(
                cnode.node_id, surface_nodes[target].node_id,
                RelationType.PREDICTION.value, count / total,
                cfg.evidence_confidence, source,
                {"count": count, "context_total": total, "context_width": len(ctx), "estimator": "mle_ngram"},
            ))

    # Repeated contiguous phrases become canonical composite observations. They are
    # not assigned a semantic tier by hand; their level follows member composition.
    phrase_counts: Counter[tuple[str, ...]] = Counter()
    for sent in sentences:
        for width in range(2, cfg.max_phrase_words + 1):
            if width > len(sent):
                break
            for i in range(len(sent)-width+1):
                phrase_counts[tuple(sent[i:i+width])] += 1
    phrase_nodes: dict[tuple[str, ...], IRNode] = {}
    for phrase, count in phrase_counts.items():
        if count < cfg.min_phrase_count:
            continue
        pnode = make_composite(
            Modality.TEXT.value,
            [surface_nodes[t] for t in phrase],
            NodeKind.COMPOSITE.value,
            label=" ".join(phrase),
        )
        # Context templates of the exact same members may have different kind/ID;
        # that is intentional: observation and predictive context are distinct roles.
        pnode.metadata = {"observed_count": count, "role": "observed_phrase"}
        phrase_nodes[phrase] = pnode
        all_nodes[pnode.node_id] = pnode

    # Contextual substitution: two words are candidates when they occupy overlapping
    # immediate (left,right) slots. This is symmetric evidence and never interpreted
    # as identity. Sentence edges use explicit sentinels only in evidence calculation.
    word_contexts: dict[str, Counter] = defaultdict(Counter)
    for sent in sentences:
        for i, word in enumerate(sent):
            left = sent[i-1] if i > 0 else "<BOS>"
            right = sent[i+1] if i+1 < len(sent) else "<EOS>"
            word_contexts[word][(left, right)] += 1

    words = sorted(word_contexts)
    substitution_candidates: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for i, a in enumerate(words):
        for b in words[i+1:]:
            score = _weighted_jaccard(word_contexts[a], word_contexts[b])
            if score >= cfg.min_substitution_score:
                substitution_candidates[a].append((score, b))
                substitution_candidates[b].append((score, a))

    emitted_pairs: set[tuple[str, str]] = set()
    for a, vals in substitution_candidates.items():
        vals.sort(key=lambda x: (-x[0], x[1]))
        for score, b in vals[:cfg.max_substitutions_per_word]:
            pair = tuple(sorted((a, b)))
            if pair in emitted_pairs:
                continue
            emitted_pairs.add(pair)
            common_slots = sum(min(word_contexts[a][k], word_contexts[b][k]) for k in set(word_contexts[a]) & set(word_contexts[b]))
            meta = {"method": "weighted_jaccard_lr_slot", "shared_slot_mass": common_slots}
            evidence.append(RelationEvidence(surface_nodes[a].node_id, surface_nodes[b].node_id, RelationType.SUBSTITUTION.value, score, cfg.evidence_confidence, source, meta))
            evidence.append(RelationEvidence(surface_nodes[b].node_id, surface_nodes[a].node_id, RelationType.SUBSTITUTION.value, score, cfg.evidence_confidence, source, meta))

    report = LanguageEvidenceReport(
        documents=len(texts),
        sentences=len(sentences),
        lexical_tokens=sum(len(s) for s in sentences),
        unique_surfaces=len(surface_nodes),
        q_nodes=sum(1 for n in all_nodes.values() if n.kind == NodeKind.QUANT.value),
        phrase_nodes=len(phrase_nodes),
        transition_evidence=sum(1 for e in evidence if e.relation_type == RelationType.TRANSITION.value),
        prediction_evidence=sum(1 for e in evidence if e.relation_type == RelationType.PREDICTION.value),
        substitution_evidence=sum(1 for e in evidence if e.relation_type == RelationType.SUBSTITUTION.value),
    )
    doc = ForgeDocument(nodes=sorted(all_nodes.values(), key=lambda n: (n.level, n.node_id)), evidence=evidence)
    return doc, report


def load_text_inputs(paths: list[str | Path]) -> list[str]:
    out = []
    for p in paths:
        out.append(Path(p).read_text(encoding="utf-8"))
    return out
