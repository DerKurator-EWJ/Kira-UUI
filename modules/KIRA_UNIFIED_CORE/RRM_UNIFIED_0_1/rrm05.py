"""RRM 0.5: one evidence table for unconditional and contextual transitions.
Research prototype: no autonomous perception, causality or general intelligence.
"""
import json
import sqlite3
from collections import Counter


def validate(s):
    if not isinstance(s, dict) or not isinstance(s.get('nodes'), dict) or not isinstance(s.get('edges'), list):
        raise ValueError('state requires nodes and edges')
    for edge in s['edges']:
        if not isinstance(edge, (list, tuple)) or len(edge) != 3 or edge[0] not in s['nodes'] or edge[2] not in s['nodes']:
            raise ValueError('invalid edge')


def neighborhood(s, index):
    a, _, b = s['edges'][index]
    def role(n):
        return 'SRC' if n == a else 'DST' if n == b else 'OTHER:' + str(s['nodes'][n])
    return tuple(sorted((role(x), str(r), role(y)) for j, (x, r, y) in enumerate(s['edges'])
                        if j != index and (x in (a, b) or y in (a, b))))


def changes(before, after):
    """Return evidence only for unambiguous, aligned transitions.
    Alignment is an explicit prototype precondition, not solved tracking.
    """
    validate(before); validate(after)
    if before['nodes'] != after['nodes'] or len(before['edges']) != len(after['edges']):
        return []
    different = [i for i, (a, b) in enumerate(zip(before['edges'], after['edges'])) if a != b]
    if len(different) > 1:
        return []
    if different:
        i = different[0]
        x, old, y = before['edges'][i]
        xx, new, yy = after['edges'][i]
        if (x, y) != (xx, yy) or old == new:
            return []
        indices = [i]
    else:
        indices = list(range(len(before['edges'])))
    out = []
    for i in indices:
        x, old, y = before['edges'][i]
        outcome = after['edges'][i][1] if different else 'NO_CHANGE'
        # Both specificity levels are projections of the same observation.
        signature = (str(before['nodes'][x]), str(old), str(before['nodes'][y]))
        out.append((signature, neighborhood(before, i), str(outcome)))
    return out


class Kernel:
    def __init__(self, path=':memory:'):
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.execute('CREATE TABLE IF NOT EXISTS observations (stream TEXT, position INTEGER, raw TEXT, PRIMARY KEY(stream,position))')
        self.db.execute('''CREATE TABLE IF NOT EXISTS evidence (
            stream TEXT, signature TEXT, context TEXT, outcome TEXT, count INTEGER NOT NULL,
            PRIMARY KEY(stream,signature,context,outcome))''')
        self.db.commit()

    def observe(self, stream, state):
        validate(state)
        raw = json.dumps(state, sort_keys=True, ensure_ascii=False)
        with self.db:
            row = self.db.execute('SELECT position,raw FROM observations WHERE stream=? ORDER BY position DESC LIMIT 1', (stream,)).fetchone()
            position = row[0] + 1 if row else 0
            if row:
                for sig, ctx, outcome in changes(json.loads(row[1]), state):
                    self.db.execute('''INSERT INTO evidence VALUES (?,?,?,?,1)
                        ON CONFLICT(stream,signature,context,outcome) DO UPDATE SET count=count+1''',
                        (stream, json.dumps(sig), json.dumps(ctx), outcome))
            self.db.execute('INSERT INTO observations VALUES (?,?,?)', (stream, position, raw))
        return position

    def predict(self, stream, state, focus=None, min_support=2, contextual=True):
        """contextual=True requires exact neighborhood; False aggregates same evidence.
        The two modes are queries, NOT two separate learned models.
        """
        validate(state)
        candidates = []
        for i, (a, relation, b) in enumerate(state['edges']):
            if focus is not None and focus != i:
                continue
            sig = json.dumps((str(state['nodes'][a]), str(relation), str(state['nodes'][b])))
            ctx = json.dumps(neighborhood(state, i))
            if contextual:
                rows = self.db.execute('SELECT outcome,count FROM evidence WHERE stream=? AND signature=? AND context=?', (stream, sig, ctx)).fetchall()
            else:
                rows = self.db.execute('SELECT outcome,SUM(count) FROM evidence WHERE stream=? AND signature=? GROUP BY outcome', (stream, sig)).fetchall()
            if not rows:
                continue
            rows.sort(key=lambda row: (-row[1], row[0]))
            total = sum(n for _, n in rows)
            if total < min_support:
                continue
            if len(rows) > 1 and rows[0][1] == rows[1][1]:
                return {'status':'OPEN', 'reason':'tied outcomes', 'alternatives':rows}
            candidates.append((i, rows, total))
        if len(candidates) != 1:
            return {'status':'OPEN', 'reason':'no unique supported target', 'candidates':len(candidates)}
        i, rows, total = candidates[0]
        outcome, support = rows[0]
        result = json.loads(json.dumps(state))
        if outcome != 'NO_CHANGE':
            result['edges'][i][1] = outcome
        return {'status':'HYPOTHESIS', 'result':result, 'support':support,
                'total':total, 'frequency':support/total, 'contextual':contextual,
                'alternatives':rows}

    def dream(self, stream, state, steps=3, **kwargs):
        """Hypothetical continuations; never writes observations or evidence."""
        results = []
        current = state
        for _ in range(steps):
            prediction = self.predict(stream, current, **kwargs)
            results.append(prediction)
            if prediction['status'] != 'HYPOTHESIS':
                break
            current = prediction['result']
        return results

    def close(self):
        self.db.close()
