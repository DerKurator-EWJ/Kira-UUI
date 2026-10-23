"""UUI 1.0: transport-neutral, semantic-neutral sensor observation ingestion.

Accepts exported quantizer results, not raw camera/microphone streams.
No object recognition, Android execution, or semantic inference is claimed.
"""
import json
import math
import sqlite3

MODALITIES={'visual':7,'audio':6}
MAX_BATCH=1000

class SensorStore:
    def __init__(self, relations):
        self.relations=relations
        self.db=relations.db
        self.db.execute('''CREATE TABLE IF NOT EXISTS uui10_sensor_events (
            event_id TEXT PRIMARY KEY, modality TEXT NOT NULL, source TEXT NOT NULL,
            sequence INTEGER NOT NULL, prototype TEXT NOT NULL,
            features_json TEXT NOT NULL, metadata_json TEXT NOT NULL,
            UNIQUE(source,modality,sequence))''')
        self.db.commit()

    def ingest(self, item):
        if not isinstance(item,dict):raise ValueError('sensor event must be object')
        required={'event_id','modality','source','sequence','prototype_id','features'}
        if not required.issubset(item):raise ValueError('missing sensor event fields: '+str(sorted(required-set(item))))
        eid=item['event_id']; modality=item['modality']; source=item['source']; prototype=item['prototype_id']; seq=item['sequence']
        for s in (eid,source):
            if not isinstance(s,str) or not s or len(s)>200:raise ValueError('invalid event/source id')
        if modality not in MODALITIES:raise ValueError('unsupported modality')
        if type(seq)!=int or seq<0:raise ValueError('sequence must be nonnegative integer')
        if type(prototype) not in (int,str) or isinstance(prototype,str) and (not prototype or len(prototype)>200):
            raise ValueError('invalid prototype id')
        features=item['features']
        if not isinstance(features,list) or len(features)!=MODALITIES[modality]:raise ValueError('invalid feature vector size')
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in features):raise ValueError('nonfinite/non-numeric feature')
        metadata=item.get('metadata',{})
        if not isinstance(metadata,dict) or len(json.dumps(metadata))>8192:raise ValueError('invalid metadata')
        if self.db.execute('SELECT 1 FROM uui10_sensor_events WHERE event_id=?',(eid,)).fetchone():
            raise ValueError('duplicate event_id')
        node_event='sensor.event:'+eid
        node_proto='sensor.prototype:'+modality+':'+source+':'+str(prototype)
        node_source='sensor.source:'+source
        relation_id='sensor.link:'+eid
        # Atomic across all tables: no half-imported observations.
        if self.db.execute('SELECT 1 FROM uui10_sensor_events WHERE source=? AND modality=? AND sequence=?',(source,modality,seq)).fetchone():
            raise ValueError('duplicate source/modality/sequence')
        with self.db:
            self.db.execute('INSERT INTO uui10_sensor_events VALUES (?,?,?,?,?,?,?)',
                (eid,modality,source,seq,str(prototype),json.dumps(features),json.dumps(metadata,sort_keys=True)))
            for nid in (node_event,node_proto,node_source):
                self.db.execute('INSERT OR IGNORE INTO uui08_nodes(id,label) VALUES (?,NULL)',(nid,))
            self.db.execute('INSERT INTO uui08_relations(id,source,type,target,status) VALUES (?,?,?,?,?)',
                (relation_id,node_event,'sensor.quantized_as',node_proto,'active'))
            self.db.execute('INSERT INTO uui08_relations(id,source,type,target,status) VALUES (?,?,?,?,?)',
                ('sensor.origin:'+eid,node_source,'sensor.produced',node_event,'active'))
            previous=self.db.execute('''SELECT event_id FROM uui10_sensor_events
                WHERE source=? AND modality=? AND sequence<? ORDER BY sequence DESC LIMIT 1''',
                (source,modality,seq)).fetchone()
            if previous:
                self.db.execute('INSERT INTO uui08_relations(id,source,type,target,status) VALUES (?,?,?,?,?)',
                    ('sensor.next:'+eid,'sensor.event:'+previous[0],'sensor.precedes',node_event,'active'))
        return {'event_id':eid,'modality':modality,'event_node':node_event,
                'prototype_node':node_proto,'relations_created':2+bool(previous)}

    def ingest_batch(self, events):
        if not isinstance(events,list) or len(events)>MAX_BATCH:raise ValueError('invalid batch')
        # Each event is atomic; successful preceding events remain on a later failure.
        return [self.ingest(event) for event in events]

    def inspect(self, event_id):
        row=self.db.execute('SELECT modality,source,sequence,prototype,features_json,metadata_json FROM uui10_sensor_events WHERE event_id=?',(event_id,)).fetchone()
        if row is None:raise ValueError('unknown sensor event')
        return {'event_id':event_id,'modality':row[0],'source':row[1],'sequence':row[2],
                'prototype_id':row[3],'features':json.loads(row[4]),'metadata':json.loads(row[5])}
