"""Strict SEM parsing and grammar; no ML dependencies.

IDs 0..76 follow Table 5. BOS/EOS are internal VQ decoder symbols only.
The datasets spell the final marker <extrude_end>; <extrusion_end> is accepted.
"""
from __future__ import annotations

import re

NAMES = ['pad', 'line', 'arc', 'circle', '<curve_end>', '<loop_end>',
         '<face_end>', '<sketch_end>', 'add', 'cut', 'intersect', '<extrude_end>', '-1']
NAMES += [str(i) for i in range(64)]
TOKEN_ID = {name: i for i, name in enumerate(NAMES)}
PAD, BOS, EOS, VOCAB_SIZE = 0, 77, 78, 79
CURVES = {1: 2, 2: 4, 3: 8}
NUMBERS = set(range(13, 77))
ROTATIONS = {12, 13, 14}


def cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def orient(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


class Grammar:
    """Token-type/arity grammar, plus numeric guards against known kernel-invalid
    degeneracies (reversed extrusion bounds, zero/parallel sketch axes, zero-length
    lines, degenerate arcs, and a non-add first operation). The guards exclude
    candidate values rather than editing decoded geometry, and never forbid every
    legal value at a position (see numeric_forbidden callers). Zero-length-line and
    degenerate-arc checks cover both consecutive curves within a loop and the
    loop-closing edge back to the loop's first curve."""

    def __init__(self, max_curves=10000, force_add=False, max_loops=10000):
        self.state = 'start'
        self.remaining = 0
        self.curves = 0
        self.max_curves = max_curves
        self.max_loops = max_loops
        self.loops = 0
        self.ext_index = 0
        self.force_add = force_add
        self.curve_type = None
        self.curve_values = []
        self.prev_curve = None
        self.first_curve = None
        self.ext_values = []

    def allowed(self):
        if self.state in ('start', 'after_curve', 'after_loop', 'after_face'):
            values = set(CURVES) if self.curves < self.max_curves else set()
            if self.state != 'after_curve' and self.loops >= self.max_loops:
                values.clear()
            if self.state == 'after_curve':
                values.add(5)
            elif self.state == 'after_loop':
                values.add(6)
            elif self.state == 'after_face':
                values.add(7)
            return values
        if self.state == 'boolean':
            return {8} if self.force_add else {8, 9, 10}
        return {
            'curve_number': NUMBERS, 'curve_end': {4},
            'ext_number': ROTATIONS if 5 <= self.ext_index < 14 else NUMBERS,
            'ext_end': {11}, 'eos': {EOS}, 'done': set(),
        }[self.state]

    def numeric_forbidden(self):
        """Values illegal at the current position beyond token-type/arity, given
        values already chosen earlier in this pair. Callers must still verify this
        does not exclude every allowed value before applying it (ties can make a
        position unavoidable; in that case fall back to unconstrained)."""
        if self.state == 'ext_number':
            if self.ext_index == 1:
                lower = self.ext_values[0]
                return {t for t in NUMBERS if t - 13 <= lower}
            if self.ext_index == 7 and self.ext_values[5] == 0 and self.ext_values[6] == 0:
                return {13}
            if self.ext_index == 13:
                x, z_prefix = self.ext_values[5:8], self.ext_values[11:13]
                return {t for t in ROTATIONS
                        if not any(z_prefix + [t - 13]) or not any(cross(x, z_prefix + [t - 13]))}
        elif self.state == 'curve_number' and len(self.curve_values) == 1 and self.prev_curve is not None:
            prev_type, prev_values = self.prev_curve
            x_chosen = self.curve_values[0]
            if prev_type == 1:
                return {13 + prev_values[1]} if x_chosen == prev_values[0] else set()
            if prev_type == 2:
                start, middle = prev_values[:2], prev_values[2:4]
                return {t for t in NUMBERS if orient(start, middle, [x_chosen, t - 13]) == 0}
        elif self.state == 'after_curve' and self.first_curve is not None and self.prev_curve is not None:
            prev_type, prev_values = self.prev_curve
            end = self.first_curve[1][:2]
            if prev_type == 1 and prev_values[:2] == end:
                return {5}
            if prev_type == 2 and orient(prev_values[:2], prev_values[2:4], end) == 0:
                return {5}
        return set()

    def advance(self, token):
        if token not in self.allowed():
            raise ValueError(f'Illegal token {token} in grammar state {self.state}')
        if self.state in ('start', 'after_curve', 'after_loop', 'after_face'):
            if token in CURVES:
                if self.state != 'after_curve':
                    self.loops += 1
                self.curves += 1
                self.remaining = CURVES[token]
                self.curve_type, self.curve_values = token, []
                self.state = 'curve_number'
            else:
                if token == 5:
                    self.prev_curve, self.first_curve = None, None
                self.state = {5: 'after_loop', 6: 'after_face', 7: 'boolean'}[token]
        elif self.state == 'curve_number':
            self.curve_values.append(token - 13)
            self.remaining -= 1
            if self.remaining == 0:
                self.state = 'curve_end'
        elif self.state == 'curve_end':
            self.prev_curve = (self.curve_type, self.curve_values)
            if self.first_curve is None:
                self.first_curve = self.prev_curve
            self.state = 'after_curve'
        elif self.state == 'boolean':
            self.state, self.ext_index = 'ext_number', 0
        elif self.state == 'ext_number':
            self.ext_values.append(token - 13)
            self.ext_index += 1
            if self.ext_index == 17:
                self.state = 'ext_end'
        elif self.state == 'ext_end':
            self.state = 'eos'
        elif self.state == 'eos':
            self.state = 'done'


def tokenize(text):
    text = text.replace('<extrusion_end>', '<extrude_end>')
    words = re.split(r'[\s,]+', text.strip())
    if not words or words == ['']:
        raise ValueError('Empty CAD sequence')
    try:
        ids = [TOKEN_ID[w] for w in words]
    except KeyError as e:
        raise ValueError(f'Unknown CAD token: {e.args[0]}') from e
    if PAD in ids:
        raise ValueError('Padding is not allowed inside a CAD sequence')
    return ids


def stringify(ids):
    text = ''
    for t in ids:
        if not 1 <= t < len(NAMES):
            raise ValueError(f'Not a semantic CAD token: {t}')
        text += (',' if t >= 12 else ' ') + NAMES[t]
    return text.strip()


def split_pairs(ids):
    pairs, pair = [], []
    grammar = Grammar()
    for token in ids:
        grammar.advance(token)
        pair.append(token)
        if token == 11:
            grammar.advance(EOS)
            pairs.append(pair)
            pair, grammar = [], Grammar()
    if pair or not pairs:
        raise ValueError('Incomplete sketch–extrusion pair')
    return pairs


def primitive_ids(pair, extrusion_partition='three', sketch_partition='curve'):
    """Versioned curve/loop pooling; trailing end markers belong to that group."""
    if sketch_partition not in ('curve', 'loop'):
        raise ValueError('Unknown sketch partition')
    split_pairs(pair)
    if pair.count(11) != 1:
        raise ValueError('Pooling expects one sketch–extrusion pair')
    result, current, ext = [], -1, None
    for i, token in enumerate(pair):
        if token in CURVES and (sketch_partition == 'curve' or current < 0 or pair[i-1] in (5, 6)):
            current += 1
        if token in (8, 9, 10):
            ext = i
            break
        result.append(current)
    if ext is None:
        raise ValueError('No extrusion')
    sizes = {'three': [6, 9, 4], 'rotation_rows': [6, 3, 3, 3, 4]}
    if extrusion_partition not in sizes:
        raise ValueError('Unknown extrusion partition')
    for length in sizes[extrusion_partition]:
        current += 1
        result.extend([current] * length)
    if len(result) != len(pair):
        raise ValueError('Extrusion must have B + 17 parameters + END')
    return result


def analyze(text, max_pair_tokens=512, max_primitives=12, extrusion_partition='three', sketch_partition='curve'):
    pairs = split_pairs(tokenize(text))
    output = []
    for pair in pairs:
        groups = primitive_ids(pair, extrusion_partition, sketch_partition)
        if len(pair) > max_pair_tokens:
            raise ValueError(f'Pair length {len(pair)} exceeds {max_pair_tokens}')
        if max(groups) + 1 > max_primitives:
            raise ValueError(f'Primitive count {max(groups)+1} exceeds {max_primitives}')
        output.append({'tokens': pair, 'groups': groups})
    return stringify([token for p in pairs for token in p]), output


def skexgen_to_sem(item):
    """Equivalent to CADFusion's released offset/layout conversion."""
    output = []
    for index in range(int(item['num_se'])):
        xy = [[int(v) - 4 for v in row] for row in item['se_xy'][index]]
        commands = [int(v) - 3 for v in item['se_cmd'][index]]
        cursor = 0
        for cmd in commands:
            if cmd in (0, 1, 2):
                points, consumed = {0: (1, 2), 1: (2, 3), 2: (4, 5)}[cmd]
                coords = [v for row in xy[cursor:cursor + points] for v in row]
                if len(coords) != points * 2:
                    raise ValueError('Truncated curve coordinates')
                output.extend([NAMES[cmd + 1]] + [str(v) for v in coords] + ['<curve_end>'])
                cursor += consumed
            elif cmd in (-1, -2, -3):
                output.append({-1: '<loop_end>', -2: '<face_end>', -3: '<sketch_end>'}[cmd])
                cursor += 1
            else:
                raise ValueError(f'Invalid SkexGen command {cmd}')
        if cursor != len(xy):
            raise ValueError('SkexGen coordinate cursor mismatch')
        ext = [int(v) for v in item['se_ext'][index]]
        if len(ext) < 18 or ext[14] not in (1, 2, 3):
            raise ValueError('Invalid SkexGen extrusion')
        output.append({1: 'add', 2: 'cut', 3: 'intersect'}[ext[14]])
        output.extend(str(v - 1) for v in ext[:5])
        output.extend(str(v - 2) for v in ext[5:14])
        output.extend(str(v - 1) for v in ext[15:18])
        output.append('<extrude_end>')
    return ' '.join(output)
