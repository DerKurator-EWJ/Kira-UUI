"""Real neural hidden-state execution without transformers.

This test trains a tiny bidirectional GRU to separate two uses of the same surface
"bank" from sentence context. The evaluation labels train only this disposable test
teacher; Forge receives hidden vectors through the generic backend boundary and does
not import the labels as canonical knowledge.
"""
from pathlib import Path
import sys, re
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

try:
    import torch
    import torch.nn as nn
except Exception:
    print('FORGE_V06_TINY_CONTEXTUAL_SKIPPED_NO_TORCH')
    raise SystemExit(0)

from equinox_forge.hidden_state import HiddenStateSample, HiddenStateConfig, load_probe_dataset, extract_contextual_hidden_evidence
from equinox_forge.validate import validate_document

torch.manual_seed(11)
PROBES=load_probe_dataset([ROOT/'examples'/'probes'/'bank_senses.jsonl'])

def toks(s): return re.findall(r"\w+",s.lower(),flags=re.UNICODE)
words=sorted({w for p in PROBES for w in toks(p.text)})
V=['<pad>']+words; IDX={w:i for i,w in enumerate(V)}
MAX=max(len(toks(p.text)) for p in PROBES)

def encode(text):
    ws=toks(text); ids=[IDX[w] for w in ws]; bank=ws.index('bank')
    ids=ids+[0]*(MAX-len(ids)); return ids,bank

X=[]; POS=[]; Y=[]
for p in PROBES:
    ids,pos=encode(p.text); X.append(ids); POS.append(pos); Y.append(0 if p.label=='finance' else 1)
X=torch.tensor(X,dtype=torch.long); POS=torch.tensor(POS); Y=torch.tensor(Y)

class TinyContext(nn.Module):
    def __init__(self):
        super().__init__(); self.emb=nn.Embedding(len(V),12,padding_idx=0); self.gru=nn.GRU(12,12,batch_first=True,bidirectional=True); self.head=nn.Linear(24,2)
    def states(self,x): return self.gru(self.emb(x))[0]
    def forward(self,x,pos):
        h=self.states(x); row=torch.arange(x.shape[0]); target=h[row,pos]; return self.head(target),target

m=TinyContext(); opt=torch.optim.Adam(m.parameters(),lr=.03); lossfn=nn.CrossEntropyLoss()
for _ in range(220):
    logits,_=m(X,POS); loss=lossfn(logits,Y); opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    pred=m(X,POS)[0].argmax(-1); assert float((pred==Y).float().mean())==1.0

class TinyBackend:
    backend_id='tiny-bidir-gru-context'
    def manifest(self): return {'backend_id':self.backend_id,'parameters':sum(p.numel() for p in m.parameters())}
    def contextual_state(self,text,target_start,target_end,layers,pooling):
        ids,pos=encode(text); x=torch.tensor([ids],dtype=torch.long)
        with torch.no_grad(): h=m.states(x)[0,pos].float()
        return [HiddenStateSample(layer=l,vector=h.tolist(),token_count=1,metadata={'target_token_index':pos}) for l in layers]

cfg=HiddenStateConfig(layers=(-1,),top_k=3,min_cosine=-1.0)
doc,report,_=extract_contextual_hidden_evidence(TinyBackend(),PROBES,'tiny-context','1',cfg)
errs=[x for x in validate_document(doc) if x.severity=='error']; assert not errs,errs
assert report.hidden_samples==6
assert report.nearest_neighbour_label_accuracy is not None
# Do not require a scientific quality threshold from a toy fixture; the test only
# requires that a real neural hidden state was executed and the metric is finite.
assert 0.0 <= report.nearest_neighbour_label_accuracy <= 1.0
print('FORGE_V06_TINY_CONTEXTUAL_OK',report.nearest_neighbour_label_accuracy)
