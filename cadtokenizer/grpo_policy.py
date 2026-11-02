"""Grammar-conditioned sampling and likelihoods for the same CAD policy."""
from .common import require_container
require_container()
import torch
from .lossless_sem import LosslessCadConstraint


def allowed_ids(codec, prefix, settings):
    allowed = LosslessCadConstraint(codec,max_pairs=settings['max_pairs']).allowed(prefix)
    if settings['max_new_tokens']-len(prefix)<36:
        allowed -= {codec.native_ids[token-1] for token in (1,2,3)}
    if not allowed:
        raise ValueError('No legal CAD continuation within the fixed budget')
    return sorted(allowed)


def sample(model, prompt, codec, settings):
    from transformers import LogitsProcessor,LogitsProcessorList
    n = len(prompt)
    class Mask(LogitsProcessor):
        def __call__(self, ids, scores):
            allowed = allowed_ids(codec,ids[0,n:].tolist(),settings)
            mask = torch.full_like(scores,-torch.inf)
            mask[:,allowed] = 0
            return scores+mask
    ids = torch.tensor([prompt],device='cuda')
    model.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        output = model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),
            do_sample=True,temperature=settings['temperature'],top_k=0,top_p=1.,
            max_new_tokens=settings['max_new_tokens'],use_cache=True,
            eos_token_id=codec.boundaries['end'],pad_token_id=128001,
            logits_processor=LogitsProcessorList([Mask()]))
    generated = output[0,n:].tolist()
    try:
        sequence = codec.decode([codec.boundaries['start']]+generated)
    except ValueError:
        sequence = None
    return {'generated_ids':generated,'sequence':sequence}


def token_logps(model, prompt, completion, codec, settings):
    if not completion:
        raise ValueError('Cannot optimize an empty sampled trajectory')
    ids = torch.tensor([prompt+completion],device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        result = model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
    values = []
    for t, token in enumerate(completion):
        allowed = allowed_ids(codec,completion[:t],settings)
        if token not in allowed:
            raise ValueError('Sampled action is outside the optimized policy support')
        # Normalize over precisely the grammar-allowed, temperature-scaled
        # distribution used for sampling, not over the full LM vocabulary.
        logits = result.logits[0,len(prompt)-1+t,allowed].float()/settings['temperature']
        values.append(logits[allowed.index(token)]-torch.logsumexp(logits,dim=0))
    return torch.stack(values)


def clipped_loss(new_logps, old_logps, advantage, epsilon):
    ratio = torch.exp(new_logps-old_logps)
    return -torch.minimum(ratio*advantage,ratio.clamp(1-epsilon,1+epsilon)*advantage).mean()


def self_test():
    from .grpo_reward import reward,advantages
    a='line,1,2'; b='line,1,4'
    assert reward(a,b,b,True)==1.25 and reward(a,b,a,True)==0
    assert reward(a,b,b,False)==-1
    assert reward(a,b,'line,1,3',True)>reward(a,b,a,True)
    v,s=advantages([0,1,2,3]); assert s>0 and abs(sum(v))<1e-8
    assert advantages([1]*4)[0]==[0]*4
    logits=torch.tensor([0.,0.,0.],requires_grad=True)
    lp=torch.log_softmax(logits[:2],dim=0)[1]
    loss=clipped_loss(lp,lp.detach(),1.,.2);loss.backward()
    assert logits.grad[1]<0 and logits.grad[0]>0 and logits.grad[2]==0
    assert abs(float(clipped_loss(torch.tensor(.5),torch.tensor(0.),1.,.2))+1.2)<1e-6
    assert abs(float(clipped_loss(torch.tensor(-.5),torch.tensor(0.),-1.,.2))-.8)<1e-6
    print('Reward controls, group advantages, masked-policy gradient and clipping tests passed.',flush=True)
