"""Lossless, explicitly framed SEM codec and generation constraints (no ML)."""
from dataclasses import dataclass

from .representation import CURVES, EOS, Grammar, NAMES, split_pairs, stringify, tokenize

SCHEMA = 'lossless-sem-v1'


def symbol_strings():
    def name(value):
        if value == '-1':
            return 'number_minus_1'
        if value.isdigit():
            return 'number_'+value
        return value.strip('<>')
    return [f'<|cad_sem_{name(value)}|>' for value in NAMES[1:]]


FRAME_STRINGS = {'start':'<|cad_sem_start|>', 'pair_end':'<|cad_sem_pair_end|>', 'end':'<|cad_sem_end|>'}


@dataclass(frozen=True)
class LosslessSemCodec:
    native_ids: tuple
    boundaries: dict

    def __post_init__(self):
        if len(self.native_ids) != len(NAMES)-1 or set(self.boundaries) != set(FRAME_STRINGS):
            raise ValueError('Incomplete lossless SEM mapping')
        ids = list(self.native_ids)+list(self.boundaries.values())
        if any(type(value) is not int or value < 0 for value in ids) or len(set(ids)) != len(ids):
            raise ValueError('SEM symbols and frames require distinct nonnegative integer IDs')

    def encode_pairs(self, sequence):
        return [[self.native_ids[token-1] for token in pair] for pair in split_pairs(tokenize(sequence))]

    def encode(self, sequence):
        result = [self.boundaries['start']]
        for pair in self.encode_pairs(sequence):
            result.extend(pair)
            result.append(self.boundaries['pair_end'])
        return result+[self.boundaries['end']]

    def decode(self, ids):
        ids = list(ids)
        if len(ids)<3 or ids[0]!=self.boundaries['start'] or ids[-1]!=self.boundaries['end']:
            raise ValueError('Missing CAD start/end frame')
        inverse = {value:index+1 for index,value in enumerate(self.native_ids)}
        output,current = [],[]
        pairs = 0
        for value in ids[1:-1]:
            if value==self.boundaries['pair_end']:
                if not current or len(split_pairs(current))!=1:
                    raise ValueError('Each frame must contain exactly one complete SEM pair')
                output.extend(current)
                current=[]
                pairs+=1
            elif value in inverse:
                current.append(inverse[value])
            else:
                raise ValueError('Unknown CAD ID or unexpected frame')
        if current or not pairs:
            raise ValueError('Missing pair end or empty CAD program')
        return stringify(output)

    def to_dict(self):
        return {'schema':SCHEMA,'semantic_names':NAMES[1:],'native_ids':list(self.native_ids),
                'boundaries':dict(self.boundaries),'symbol_strings':symbol_strings(),'frame_strings':FRAME_STRINGS}

    @classmethod
    def from_dict(cls, data):
        if data['schema']!=SCHEMA or data['semantic_names']!=NAMES[1:]:
            raise ValueError('Unknown lossless SEM schema or alphabet')
        return cls(tuple(data['native_ids']),dict(data['boundaries']))


def install_codec(tokenizer):
    """Add atomic CAD symbols in memory; no model weights are loaded or resized."""
    strings = symbol_strings()
    tokenizer.add_special_tokens({'additional_special_tokens':strings+list(FRAME_STRINGS.values())})
    mapping = [int(tokenizer.convert_tokens_to_ids(value)) for value in strings]
    boundaries = {key:int(tokenizer.convert_tokens_to_ids(value)) for key,value in FRAME_STRINGS.items()}
    for value,token in zip(strings+list(FRAME_STRINGS.values()),mapping+list(boundaries.values())):
        if tokenizer.encode(value,add_special_tokens=False)!=[token]:
            raise ValueError(f'CAD symbol is not atomic: {value}')
    return LosslessSemCodec(tuple(mapping),boundaries)


class LosslessCadConstraint:
    """Allowed next IDs after the prompt's CAD-start token.

    Stop generation on CAD-end. These grammar and numeric guards constrain
    syntax; they do not prove that the resulting CAD solid is valid.
    """
    def __init__(self, codec, max_pairs=64, numeric_guards=True):
        if max_pairs<1:
            raise ValueError('Need a positive pair limit')
        self.codec,self.max_pairs,self.numeric_guards = codec,max_pairs,numeric_guards

    def allowed(self, generated):
        inverse = {value:index+1 for index,value in enumerate(self.codec.native_ids)}
        grammar = Grammar(force_add=True)
        pairs = 0
        between = False
        terminated = False
        for value in generated:
            if terminated:
                raise ValueError('Tokens after CAD-end')
            if value==self.codec.boundaries['end']:
                if not between:
                    raise ValueError('CAD-end before a complete pair')
                terminated=True
            elif value==self.codec.boundaries['pair_end']:
                if grammar.state!='eos' or between:
                    raise ValueError('Pair boundary inside an incomplete pair')
                grammar.advance(EOS)
                pairs+=1
                between=True
            elif value in inverse:
                if between:
                    if pairs>=self.max_pairs:
                        raise ValueError('Pair limit exceeded')
                    grammar=Grammar(force_add=False)
                    between=False
                grammar.advance(inverse[value])
            else:
                raise ValueError('Unexpected CAD token')
        if terminated:
            return set()
        if between:
            result = {self.codec.boundaries['end']}
            if pairs<self.max_pairs:
                result.update(self.codec.native_ids[token-1] for token in CURVES)
            return result
        if grammar.state=='eos':
            return {self.codec.boundaries['pair_end']}
        allowed = grammar.allowed()
        if self.numeric_guards:
            forbidden = grammar.numeric_forbidden() & allowed
            if forbidden != allowed:
                allowed = allowed-forbidden
        return {self.codec.native_ids[token-1] for token in allowed}
