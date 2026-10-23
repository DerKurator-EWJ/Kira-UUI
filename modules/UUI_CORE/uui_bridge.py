"""UUI bridge 0.1: shared protocol, not a programming language or merged VM."""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor'/'kira'))
sys.path.insert(0,str(HERE/'vendor'/'forge'))
from uui_relations import RelationStore
from numeric_core import NumericCore
from evidence_layer import EvidenceLayer
from inquiry_controller import InquiryController
from experiment_planner import ExperimentPlanner
from equinox_forge.unified_numeric_kernel import UnifiedNumericExecutionKernel, load_unified_numeric_kernel_pack

@dataclass(frozen=True)
class ScopedIdentity:
    namespace: str
    value: int
    def __post_init__(self):
        if not self.namespace or not isinstance(self.value,int) or isinstance(self.value,bool):
            raise ValueError('invalid scoped identity')
    def key(self):
        return f'{self.namespace}:{self.value}'

class UUIBridge:
    """Forge knowledge is read-only; Kira evidence is mutable. IDs are never conflated."""
    def __init__(self, *, forge_pack_path=None, kira_db=':memory:'):
        self.numeric=NumericCore(kira_db)
        self.evidence=EvidenceLayer(self.numeric.db)
        self.relations=RelationStore(self.numeric.db)
        self.inquiry=InquiryController(self.evidence)
        self.planner=ExperimentPlanner(self.inquiry)
        self.forge=(UnifiedNumericExecutionKernel(load_unified_numeric_kernel_pack(forge_pack_path))
                    if forge_pack_path else None)
    def numeric_observation(self, value, *, source='observed'):
        item_id=self.numeric.observe(value,source=source)
        val=self.numeric.value(item_id)
        return {'id':ScopedIdentity('kira.numeric',item_id).key(),
                'rational':[str(val.numerator),str(val.denominator)]}
    def forge_query(self, subject_id, relation_id):
        if self.forge is None:raise RuntimeError('forge pack not loaded')
        result=self.forge.resolve(subject_id,relation_id)
        return {'subject':ScopedIdentity('forge.subject',int(subject_id)).key(),
                'relation':ScopedIdentity('forge.relation',int(relation_id)).key(),
                'resolution':result.to_dict(), 'source':'forge_pack'}
    def record_forge_resolution(self, *, event_id, subject_id, relation_id):
        """Store Forge output as an external claim, never as an observed fact."""
        q=self.forge_query(subject_id,relation_id)
        r=q['resolution']
        if r['status']!='resolved':
            return {'recorded':False,'status':r['status']}
        payload={'status':r['status'],'values':r['values']}
        result=self.evidence.record(event_id=event_id,subject=q['subject'],
            predicate=q['relation'],value=payload,context={'source':'forge_pack'},
            source_type='external_claim',source_ref='forge:pack',confidence=1.0)
        return {'recorded':True,**result}
    def observe_claim(self, *, event_id, subject, predicate, value, context=None,source_ref='user'):
        result=self.evidence.record(event_id=event_id,subject=subject,predicate=predicate,
            value=value,context=context,source_type='observation',source_ref=source_ref)
        self.inquiry.sync_reviews()
        return result
    def ask_about(self, subject, predicate):
        return self.inquiry.ask_about(subject, predicate)
    def register_capability(self, capability_id, predicate, description, **kwargs):
        return self.planner.register_capability(capability_id, predicate, description, **kwargs)
    def propose_experiments(self, inquiry_id, *, limit=5):
        return self.planner.propose(inquiry_id, limit=limit)
    def approve_experiment(self, experiment_id):
        return self.planner.approve(experiment_id)
    def record_verified_experiment(self, experiment_id, *, event_id, value, source_ref, **kwargs):
        return self.planner.complete(experiment_id, event_id=event_id, value=value, source_ref=source_ref, **kwargs)
    def close(self):self.numeric.close()
