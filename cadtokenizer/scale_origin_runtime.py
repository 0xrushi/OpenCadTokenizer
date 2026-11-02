"""Batched inference for the unchanged VQ model, inside the cached image only."""
from .common import require_container


def encode_pairs(model,pairs,batch_size):
    require_container()
    import torch
    from .vq import collate_pairs
    result=[None]*len(pairs)
    ordered=sorted(range(len(pairs)),key=lambda i:(len(pairs[i]['tokens']),i))
    with torch.inference_mode():
        for offset in range(0,len(ordered),batch_size):
            indices=ordered[offset:offset+batch_size]
            batch=collate_pairs([pairs[i] for i in indices])
            _,codes,mask=model.encode(batch['tokens'].cuda(),batch['groups'].cuda())
            for index,values,valid in zip(indices,codes.tolist(),mask.tolist()):
                result[index]=[v for v,keep in zip(values,valid) if keep]
    return result


def _batch_decode(model,codes,first_flags):
    import torch
    from .representation import BOS,CURVES,EOS,Grammar
    width=model.config['vq']['width'];count=max(map(len,codes));device=model.codebook.weight.device
    memory=torch.zeros((len(codes),count,width),device=device,dtype=model.codebook.weight.dtype)
    masks=torch.zeros((len(codes),count),device=device,dtype=torch.bool)
    for i,ids in enumerate(codes):
        memory[i,:len(ids)]=model.codebook.weight[torch.tensor(ids,device=device)];masks[i,:len(ids)]=True
    max_length=model.config['data']['max_pair_tokens']+1
    extrusions=3 if model.config['data']['extrusion_partition']=='three' else 5
    budget=model.config['data']['max_primitives']-extrusions
    if budget<1 or max_length<CURVES[1]+25:raise ValueError('Invalid primitive budget')
    loop=model.config['data'].get('sketch_partition','curve')=='loop'
    grammars=[Grammar(max_curves=(max_length-23)//4 if loop else budget,
        max_loops=model.config['data']['max_primitives']-extrusions if loop else 10000,force_add=flag) for flag in first_flags]
    prefixes=[[BOS] for _ in codes];results=[None]*len(codes);active=list(range(len(codes)))
    for _ in range(max_length):
        if not active:break
        ids=torch.tensor([prefixes[i] for i in active],device=device)
        selection=torch.tensor(active,device=device)
        logits=model.decode(ids,memory[selection],masks[selection])[:,-1].float()
        allowed_mask=torch.full((len(active),logits.shape[-1]),-torch.inf)
        bad=set()
        for row,i in enumerate(active):
            allowed=sorted(grammars[i].allowed())
            remaining=max_length-len(prefixes[i])+1
            allowed=[t for t in allowed if t not in CURVES or remaining>=CURVES[t]+25]
            if not allowed:
                results[i]={'tokens':None,'error':'No grammar continuation fits the remaining token budget'};bad.add(i)
                continue
            forbidden=grammars[i].numeric_forbidden()&set(allowed)
            if len(forbidden)==len(allowed):forbidden=set()
            allowed_mask[row,[t for t in allowed if t not in forbidden]]=0
        chosen=(logits+allowed_mask.to(device)).argmax(-1).tolist();following=[]
        for i,token in zip(active,chosen):
            if i in bad:continue
            grammars[i].advance(token)
            if token==EOS:results[i]={'tokens':prefixes[i][1:],'error':None}
            else:prefixes[i].append(token);following.append(i)
        active=following
    for i in active:results[i]={'tokens':None,'error':'CAD reconstruction reached maximum length before EOS'}
    return results


def decode_pairs(model,codes,first_flags,batch_size):
    require_container()
    import torch
    results=[None]*len(codes)
    # Sorting uses only the stream's code counts, never original SEM or metadata.
    order=sorted(range(len(codes)),key=lambda i:(len(codes[i]),i))
    with torch.inference_mode():
        for offset in range(0,len(order),batch_size):
            chosen=order[offset:offset+batch_size]
            if batch_size==1:
                i=chosen[0]
                try:
                    memory=model.codebook.weight[torch.tensor(codes[i],device='cuda')]
                    results[i]={'tokens':model.reconstruct(memory,constrained=True,force_add=first_flags[i]),'error':None}
                except ValueError as exc:results[i]={'tokens':None,'error':str(exc)}
            else:
                rows=_batch_decode(model,[codes[i] for i in chosen],[first_flags[i] for i in chosen])
                for i,row in zip(chosen,rows):results[i]=row
    return results
