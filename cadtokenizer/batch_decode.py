"""Incremental SEM grammar and batched greedy decoding with left padding."""
from .lossless_sem import LosslessCadConstraint
from .representation import Grammar,CURVES,EOS


class IncrementalConstraint:
    def __init__(self,codec,max_pairs):
        self.codec=codec;self.max_pairs=max_pairs
        self.inverse={v:i+1 for i,v in enumerate(codec.native_ids)}
        self.grammar=Grammar(force_add=True);self.pairs=0;self.between=False;self.terminated=False

    def advance(self,value):
        if self.terminated:raise ValueError('Tokens after CAD-end')
        if value==self.codec.boundaries['end']:
            if not self.between:raise ValueError('CAD-end before complete pair')
            self.terminated=True
        elif value==self.codec.boundaries['pair_end']:
            if self.grammar.state!='eos' or self.between:raise ValueError('Incomplete pair boundary')
            self.grammar.advance(EOS);self.pairs+=1;self.between=True
        elif value in self.inverse:
            if self.between:
                if self.pairs>=self.max_pairs:raise ValueError('Pair limit exceeded')
                self.grammar=Grammar(force_add=False);self.between=False
            self.grammar.advance(self.inverse[value])
        else:raise ValueError('Unexpected CAD token')

    def allowed(self):
        if self.terminated:return set()
        if self.between:
            result={self.codec.boundaries['end']}
            if self.pairs<self.max_pairs:result.update(self.codec.native_ids[t-1] for t in CURVES)
            return result
        if self.grammar.state=='eos':return {self.codec.boundaries['pair_end']}
        allowed=self.grammar.allowed();forbidden=self.grammar.numeric_forbidden()&allowed
        if forbidden!=allowed:allowed=allowed-forbidden
        return {self.codec.native_ids[t-1] for t in allowed}


def generate_many(model,prompts,codec,settings):
    from .common import require_container
    require_container()
    import torch
    from transformers import LogitsProcessor,LogitsProcessorList
    if not prompts:return []
    if any(not p or len(p)+settings['max_new_tokens']>settings['context_length'] for p in prompts):
        raise ValueError('Prompt exceeds inference context budget')
    width=max(map(len,prompts));pad=128001
    ids=torch.full((len(prompts),width),pad,dtype=torch.long,device='cuda')
    attention=torch.zeros_like(ids)
    for row,prompt in enumerate(prompts):
        ids[row,-len(prompt):]=torch.tensor(prompt,device='cuda');attention[row,-len(prompt):]=1
    class Mask(LogitsProcessor):
        def __init__(self):
            self.states=[IncrementalConstraint(codec,settings['max_pairs']) for _ in prompts]
            self.seen=0
            self.vocab=list(codec.native_ids)+[codec.boundaries['pair_end'],codec.boundaries['end']]
            self.index={v:i for i,v in enumerate(self.vocab)}
            self.device_ids=torch.tensor(self.vocab,device='cuda')

        def __call__(self,input_ids,scores):
            count=input_ids.shape[1]-width
            if count<self.seen:raise ValueError('Batched decoder unexpectedly rewound')
            # One small GPU-to-CPU transfer for newly generated tokens only.
            if count>self.seen:
                additions=input_ids[:,width+self.seen:].tolist()
                for state,values in zip(self.states,additions):
                    for value in values:
                        if not state.terminated:state.advance(value)
                self.seen=count
            legal=[]
            for state in self.states:
                if state.terminated:allowed={codec.boundaries['end']}
                else:
                    allowed=state.allowed()
                    if settings['max_new_tokens']-count<36:
                        allowed-={codec.native_ids[t-1] for t in CURVES}
                if not allowed:raise ValueError('No legal CAD continuation within budget')
                legal.append([v in allowed for v in self.vocab])
            legal=torch.tensor(legal,dtype=torch.bool,device=scores.device)
            selected=scores.index_select(1,self.device_ids).masked_fill(~legal,-torch.inf)
            output=torch.full_like(scores,-torch.inf)
            output.index_copy_(1,self.device_ids,selected)
            return output
    model.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        output=model.generate(input_ids=ids,attention_mask=attention,do_sample=False,
            max_new_tokens=settings['max_new_tokens'],use_cache=True,
            eos_token_id=codec.boundaries['end'],pad_token_id=pad,
            logits_processor=LogitsProcessorList([Mask()]))
    result=[]
    for values in output[:,width:].tolist():
        if codec.boundaries['end'] in values:values=values[:values.index(codec.boundaries['end'])+1]
        row={'generated_ids':values,'sequence':None,'error':None}
        try:row['sequence']=codec.decode([codec.boundaries['start']]+values)
        except ValueError as exc:row['error']=str(exc)
        result.append(row)
    return result
