"""Execute a real locally-trained causal teacher, compile its distributions, replay them."""
from pathlib import Path
import sys, tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

try:
    import torch
    import torch.nn as nn
except Exception:
    print('FORGE_V17_TINY_RETENTION_SKIPPED_NO_TORCH')
    raise SystemExit(0)

from equinox_forge.teacher_behavior import TeacherCandidate, BehaviourProbeConfig
from equinox_forge.runtime_cognition import RuntimeCognitionPolicy
from equinox_forge.behaviour_retention import compile_and_benchmark_teacher_behaviour

torch.manual_seed(17)
VOCAB=['<pad>','der','die','hund','katze','ist','ein','tier','kann','bellen','miauen','laufen']
IDX={w:i for i,w in enumerate(VOCAB)}
CORPUS=[
 ['der','hund','ist','ein','tier'],
 ['die','katze','ist','ein','tier'],
 ['der','hund','kann','bellen'],
 ['die','katze','kann','miauen'],
]*12

class TinyGRU(nn.Module):
    def __init__(self):
        super().__init__(); self.emb=nn.Embedding(len(VOCAB),10); self.gru=nn.GRU(10,18,batch_first=True); self.out=nn.Linear(18,len(VOCAB))
    def forward(self,x):
        h,_=self.gru(self.emb(x)); return self.out(h)

model=TinyGRU(); opt=torch.optim.Adam(model.parameters(),lr=0.035); loss_fn=nn.CrossEntropyLoss()
for _ in range(55):
    for words in CORPUS:
        ids=torch.tensor([[IDX[w] for w in words]],dtype=torch.long)
        logits=model(ids[:,:-1]); loss=loss_fn(logits.reshape(-1,len(VOCAB)),ids[:,1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()

class TinyBackend:
    backend_id='tiny-v017-gru'
    def manifest(self): return {'backend_id':self.backend_id,'parameters':sum(p.numel() for p in model.parameters())}
    def next_token(self,prompt,top_k,temperature):
        ids=torch.tensor([[IDX[w] for w in prompt.lower().split()]],dtype=torch.long)
        with torch.no_grad():
            logits=model(ids)[0,-1].float(); probs=torch.softmax(logits/temperature,dim=-1); vals,idxs=torch.topk(probs,min(top_k,len(VOCAB)))
        return [TeacherCandidate(int(i),VOCAB[int(i)],VOCAB[int(i)],float(p),float(logits[int(i)])) for p,i in zip(vals,idxs)]

prompts=['der hund ist ein','der hund kann','die katze kann']
with tempfile.TemporaryDirectory() as td:
    report=compile_and_benchmark_teacher_behaviour(
        TinyBackend(), prompts, Path(td)/'retention','tiny-v017','1',
        probe_cfg=BehaviourProbeConfig(top_k=4,min_probability=0.0,evidence_confidence=0.95,include_punctuation_only=True),
        cognition_policy=RuntimeCognitionPolicy(max_recall_nodes=64,max_recall_depth=2,max_predictions=4,min_prediction_score=0.000001),
        runtime_top_k=4,
    )
    assert report.top1_retention == 1.0, report.to_dict()
    assert report.direct_replay['mean_teacher_topk_recall'] == 1.0, report.to_dict()
    assert report.direct_replay['mean_ndcg_at_k'] > 0.999
    assert report.direct_replay['mean_conditional_js_divergence'] < 1e-7, report.to_dict()
    assert all(x.teacher_top1 == x.runtime_top1 for x in report.cases_detail)

print('FORGE_V17_TINY_RETENTION_OK')
