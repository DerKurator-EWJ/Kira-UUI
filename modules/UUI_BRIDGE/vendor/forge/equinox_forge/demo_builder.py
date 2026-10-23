from __future__ import annotations
from .canonical import make_text_surface, make_composite
from .ir import ForgeDocument, RelationEvidence, RelationType, SourceRef

def build_demo() -> ForgeDocument:
    nodes={}
    def word(text):
        qs,w=make_text_surface(text)
        for q in qs: nodes[q.node_id]=q
        nodes[w.node_id]=w
        return w
    hund=word("Hund"); ist=word("ist"); ein=word("ein"); tier=word("Tier"); hunde=word("Hunde")
    sentence=make_composite("text",[hund,ist,ein,tier],label="Hund ist ein Tier")
    nodes[sentence.node_id]=sentence
    ev=[
      RelationEvidence(hund.node_id,tier.node_id,RelationType.IS_A.value,0.91,0.90,SourceRef("teacher-A","1","controlled_prompt")),
      RelationEvidence(hund.node_id,tier.node_id,RelationType.IS_A.value,0.87,0.85,SourceRef("teacher-B","4","embedding+prompt")),
      RelationEvidence(hund.node_id,hunde.node_id,RelationType.SEMANTIC_SIMILARITY.value,0.96,0.95,SourceRef("teacher-A","1","context_similarity")),
    ]
    return ForgeDocument(nodes=list(nodes.values()),evidence=ev)
