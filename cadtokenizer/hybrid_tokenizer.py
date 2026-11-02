"""Public SEM-to-packet-to-SEM API for exact scale/origin metadata.

The shared decoder checkpoint is required, as for the original learned tokenizer.
The decoder does not accept source SEM or encoder state.
"""
from pathlib import Path

from .common import require_container
from .representation import analyze,tokenize
from .scale_origin_codec import HYBRID,LOSSLESS,VQ,pack,unpack,decode,metadata_for_pair


class HybridTokenizer:
    def __init__(self,checkpoint,batch_size=256):
        require_container()
        import torch
        from .vq import load_vq
        self.model,_=load_vq(checkpoint,torch.device('cuda'))
        self.model.requires_grad_(False)
        self.codebook_size=self.model.config['vq']['codebook_size']
        self.batch_size=batch_size
        if batch_size<1:raise ValueError('Batch size must be positive')

    def encode(self,sequence):
        from .scale_origin_runtime import encode_pairs
        data=self.model.config['data']
        _,pairs=analyze(sequence,data['max_pair_tokens'],data['max_primitives'],data['extrusion_partition'],data.get('sketch_partition','curve'))
        codes=encode_pairs(self.model,pairs,self.batch_size)
        return pack(HYBRID,self.codebook_size,codes,[metadata_for_pair(p['tokens']) for p in pairs])

    def decode(self,packet):
        from .scale_origin_runtime import decode_pairs
        parsed=unpack(packet)
        if parsed['kind']==LOSSLESS:return decode(packet)
        if parsed['codebook_size']!=self.codebook_size:raise ValueError('Wrong decoder codebook')
        flags=[i==0 for i in range(len(parsed['pairs']))]
        predicted=decode_pairs(self.model,parsed['pairs'],flags,self.batch_size)
        index={(tuple(codes),first):i for i,(codes,first) in enumerate(zip(parsed['pairs'],flags))}
        def callback(codes,first):
            row=predicted[index[(tuple(codes),first)]]
            if row['error']:raise ValueError(row['error'])
            return row['tokens']
        return decode(packet,callback,self.codebook_size)


def run_file(mode,source,output,checkpoint):
    from .scale_origin_experiment import numerical_context
    if not source or not output:raise ValueError('Source and output paths are required')
    if Path(source).resolve()==Path(output).resolve():raise ValueError('Output cannot overwrite source')
    with numerical_context():
        tokenizer=HybridTokenizer(checkpoint)
        Path(output).parent.mkdir(exist_ok=True,parents=True)
        if mode=='encode':Path(output).write_bytes(tokenizer.encode(Path(source).read_text()))
        elif mode=='decode':Path(output).write_text(tokenizer.decode(Path(source).read_bytes())+'\n')
        else:raise ValueError('Unknown mode')
