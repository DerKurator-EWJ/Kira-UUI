"""Real causal-model execution without requiring transformers.

This integration test trains a tiny GRU teacher locally, then exposes its next-token
probabilities through the same Forge backend boundary. It proves that v0.5 consumes
an actual computed model distribution, not only scripted fixtures.
"""
from pathlib import Path
import sys, math
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

try:
    import torch
    import torch.nn as nn
except Exception:
    print("FORGE_V05_TINY_CAUSAL_SKIPPED_NO_TORCH")
    raise SystemExit(0)

from equinox_forge.teacher_behavior import TeacherCandidate, BehaviourProbeConfig, extract_teacher_behaviour
from equinox_forge.validate import validate_document


torch.manual_seed(7)
VOCAB=["<pad>","der","die","hund","katze","ist","ein","tier","kann","bellen","miauen"]
IDX={w:i for i,w in enumerate(VOCAB)}
CORPUS=[
    ["der","hund","ist","ein","tier"],
    ["die","katze","ist","ein","tier"],
    ["der","hund","kann","bellen"],
    ["die","katze","kann","miauen"],
]*20

class TinyGRU(nn.Module):
    def __init__(self):
        super().__init__(); self.emb=nn.Embedding(len(VOCAB),12); self.gru=nn.GRU(12,20,batch_first=True); self.out=nn.Linear(20,len(VOCAB))
    def forward(self,x):
        h,_=self.gru(self.emb(x)); return self.out(h)

model=TinyGRU(); opt=torch.optim.Adam(model.parameters(),lr=0.03); loss_fn=nn.CrossEntropyLoss()
for _ in range(80):
    total=0.0
    for words in CORPUS:
        ids=torch.tensor([[IDX[w] for w in words]],dtype=torch.long)
        logits=model(ids[:,:-1]); loss=loss_fn(logits.reshape(-1,len(VOCAB)),ids[:,1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step(); total += float(loss.item())

class TinyBackend:
    backend_id="tiny-torch-gru-causal-lm"
    def manifest(self): return {"backend_id":self.backend_id,"parameters":sum(p.numel() for p in model.parameters())}
    def next_token(self,prompt,top_k,temperature):
        words=prompt.lower().split(); ids=torch.tensor([[IDX[w] for w in words]],dtype=torch.long)
        with torch.no_grad():
            logits=model(ids)[0,-1].float(); probs=torch.softmax(logits/temperature,dim=-1); vals,idxs=torch.topk(probs,min(top_k,len(VOCAB)))
        return [TeacherCandidate(int(i),VOCAB[int(i)],VOCAB[int(i)],float(p),float(logits[int(i)])) for p,i in zip(vals,idxs)]

backend=TinyBackend()
doc,report,_=extract_teacher_behaviour(backend,["der hund ist ein","die katze kann"],"tiny-gru","1",BehaviourProbeConfig(top_k=3,min_probability=0.0))
errors=[i for i in validate_document(doc) if i.severity=="error"]
assert not errors, errors
assert report.prediction_evidence == 6
# The learned teacher should put the corpus continuations at rank 1.
def top_for(prompt): return backend.next_token(prompt,1,1.0)[0].surface
assert top_for("der hund ist ein") == "tier", top_for("der hund ist ein")
assert top_for("die katze kann") == "miauen", top_for("die katze kann")
print("FORGE_V05_TINY_CAUSAL_OK")
