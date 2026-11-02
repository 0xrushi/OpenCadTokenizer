"""Source-anchored changed-number masks and normalized weighted causal CE."""
from .representation import stringify,tokenize,split_pairs


def command_origins(a,b):
    n,m=len(a),len(b)
    dp=[list(range(m+1))]+[[i]+[0]*m for i in range(1,n+1)]
    for i in range(1,n+1):
        for j in range(1,m+1):
            dp[i][j]=min(dp[i-1][j-1]+(a[i-1]!=b[j-1]),dp[i-1][j]+1,dp[i][j-1]+1)
    origins=[None]*m;i,j=n,m
    while i or j:
        if i and j and dp[i][j]==dp[i-1][j-1]+(a[i-1]!=b[j-1]):
            origins[j-1]=i-1;i-=1;j-=1
        elif i and dp[i][j]==dp[i-1][j]+1:i-=1
        else:j-=1
    return origins


def changed_numbers(source,target):
    a=stringify(tokenize(source)).split();b=stringify(tokenize(target)).split()
    origins=command_origins(a,b);flags=[]
    for command,origin in zip(b,origins):
        values=command.split(',');previous=a[origin].split(',') if origin is not None else []
        flags.append(False)
        for field,value in enumerate(values[1:],1):
            flags.append(not(previous and previous[0]==values[0] and field<len(previous) and previous[field]==value))
    if len(flags)!=len(tokenize(target)):raise ValueError('Numeric mask does not cover target SEM')
    return flags


def framed_changed_numbers(source,target):
    flags=changed_numbers(source,target);result=[];offset=0
    for pair in split_pairs(tokenize(target)):
        result.extend(flags[offset:offset+len(pair)]);offset+=len(pair)
        result.append(False) # pair boundary
    return result+[False,False] # CAD-end and language EOS


def attach_weights(sample,flags,multiplier):
    if multiplier<1:raise ValueError('Numeric multiplier must be at least one')
    prefix=sample['prompt_length']
    if len(sample['input_ids'])-prefix!=len(flags):raise ValueError('Framing/label mask length mismatch')
    if any(label!=-100 for label in sample['labels'][:prefix]):raise ValueError('Prompt labels must be masked')
    return {**sample,'weights':[0.]*prefix+[float(multiplier) if flag else 1. for flag in flags]}


def collate_weighted(samples):
    from .sft import InstructionCollator
    import torch
    batch=InstructionCollator(128001)(samples)
    weights=torch.zeros_like(batch['input_ids'],dtype=torch.float32)
    for i,sample in enumerate(samples):weights[i,:len(sample['weights'])]=torch.tensor(sample['weights'])
    return {**batch,'weights':weights}


def weighted_numerator(logits,labels,weights):
    import torch.nn.functional as F
    if labels.shape!=weights.shape or logits.shape[:2]!=labels.shape:raise ValueError('Weighted CE shapes differ')
    active=labels[:,1:].ne(-100)
    # Compute the vocabulary softmax only at supervised target positions.
    values=logits[:,:-1][active].float()
    losses=F.cross_entropy(values,labels[:,1:][active],reduction='none')
    return (losses*weights[:,1:][active]).sum()
