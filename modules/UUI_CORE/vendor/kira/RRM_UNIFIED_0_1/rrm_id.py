"""RRM-ID 0.1: hybrid tagged records, experimental wire format.
Header: bits 7..5 kind, 4..3 mode, 2..0 bit-length class.
Class 0..6 = [0,1,3,7,13,37,64] payload bits; class 7 = ULEB128 bit length.
Payload is little-endian bit-packed, zero padding required. Modes: 0 inline,
1 reference (opaque integer), 2 tuple of references (count + unsigned ULEBs),
3 reserved. No claims of self-contained arbitrary relations.
"""
from dataclasses import dataclass

WIDTHS = (0, 1, 3, 7, 13, 37, 64)
KIND_NODE, KIND_RELATION, KIND_STATE, KIND_TRANSITION, KIND_PROCESS, KIND_BOUND = range(6)
INLINE, REFERENCE, COMPOSITE = range(3)


def uleb(n):
    if not isinstance(n, int) or n < 0:
        raise ValueError('nonnegative integer required')
    result = bytearray()
    while True:
        b = n & 127
        n >>= 7
        result.append(b | (128 if n else 0))
        if not n:
            return bytes(result)


def read_uleb(data, offset):
    n = 0
    for i in range(10):
        if offset >= len(data):
            raise ValueError('truncated varint')
        b = data[offset]; offset += 1
        n |= (b & 127) << (7*i)
        if not (b & 128):
            if len(uleb(n)) != i+1:
                raise ValueError('noncanonical varint')
            return n, offset
    raise ValueError('varint too long')


@dataclass(frozen=True)
class Record:
    kind: int
    mode: int
    value: int = 0
    bits: int = 0
    refs: tuple = ()


def encode(record):
    if not 0 <= record.kind <= 5 or record.mode not in (INLINE, REFERENCE, COMPOSITE):
        raise ValueError('invalid kind/mode')
    if record.mode == COMPOSITE:
        if record.bits or record.value or any(not isinstance(x, int) or x < 0 for x in record.refs):
            raise ValueError('invalid composite')
        payload = uleb(len(record.refs)) + b''.join(uleb(x) for x in record.refs)
        bits = len(payload)*8
        value = int.from_bytes(payload, 'little')
    else:
        if record.refs or record.bits < 0 or record.bits > 1_000_000 or record.value < 0 or record.value.bit_length() > record.bits:
            raise ValueError('invalid payload')
        if record.mode == REFERENCE and record.bits == 0:
            raise ValueError('empty reference')
        bits, value = record.bits, record.value
    cls = WIDTHS.index(bits) if bits in WIDTHS else 7
    header = (record.kind << 5) | (record.mode << 3) | cls
    return bytes([header]) + (uleb(bits) if cls == 7 else b'') + value.to_bytes((bits+7)//8, 'little')


def decode(data, offset=0):
    if offset >= len(data):
        raise ValueError('missing header')
    head = data[offset]; offset += 1
    kind, mode, cls = head >> 5, (head >> 3) & 3, head & 7
    if kind > 5 or mode == 3:
        raise ValueError('invalid header')
    bits = WIDTHS[cls] if cls != 7 else None
    if bits is None:
        bits, offset = read_uleb(data, offset)
    if bits > 1_000_000:
        raise ValueError('oversized record')
    size = (bits+7)//8
    if len(data) < offset+size:
        raise ValueError('truncated payload')
    value = int.from_bytes(data[offset:offset+size], 'little'); offset += size
    if value.bit_length() > bits:
        raise ValueError('nonzero padding')
    if cls == 7 and bits in WIDTHS:
        raise ValueError('noncanonical length class')
    if mode == COMPOSITE:
        if bits % 8:
            raise ValueError('misaligned composite')
        raw = value.to_bytes(size, 'little')
        count, pos = read_uleb(raw, 0)
        if count > 100000:
            raise ValueError('too many references')
        refs = []
        for _ in range(count):
            ref, pos = read_uleb(raw, pos); refs.append(ref)
        if pos != len(raw):
            raise ValueError('trailing composite bytes')
        return Record(kind, mode, refs=tuple(refs)), offset
    if mode == REFERENCE and not bits:
        raise ValueError('empty reference')
    return Record(kind, mode, value, bits), offset


class Arena:
    """Append-only immutable records. Handles are stable; values can be shared."""
    def __init__(self):
        self.data = bytearray()
        self.offsets = []
        self.intern = {}

    def put(self, record):
        wire = encode(record)
        if wire in self.intern:
            return self.intern[wire]
        handle = len(self.offsets)
        self.offsets.append(len(self.data))
        self.data.extend(wire)
        self.intern[wire] = handle
        return handle

    def get(self, handle):
        if not isinstance(handle, int) or handle < 0 or handle >= len(self.offsets):
            raise KeyError(handle)
        record, end = decode(self.data, self.offsets[handle])
        if end != (self.offsets[handle+1] if handle+1 < len(self.offsets) else len(self.data)):
            raise ValueError('arena corruption')
        return record

    def relation(self, source, relation_type, target):
        for h in (source, relation_type, target):
            self.get(h)
        return self.put(Record(KIND_RELATION, COMPOSITE, refs=(source, relation_type, target)))


class ProcessTable:
    """Mutable process instances reference immutable snapshots; not encoded into IDs."""
    def __init__(self, arena):
        self.arena = arena
        self.current = {}
        self.next_id = 0

    def start(self, snapshot):
        self.arena.get(snapshot)
        pid = self.next_id; self.next_id += 1
        self.current[pid] = snapshot
        return pid

    def advance(self, pid, snapshot):
        self.arena.get(snapshot)
        if pid not in self.current:
            raise KeyError(pid)
        self.current[pid] = snapshot
