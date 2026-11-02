"""Explicit CAD task routing across adapters on one shared frozen backbone."""
from pathlib import Path
from .common import require_container,read_config,file_hash


class TaskExpertGenerator:
    def __init__(self,descriptor):
        require_container()
        from transformers import AutoTokenizer
        from .lossless_sft import build_model
        from .lossless_sem import LosslessSemCodec
        self.descriptor=read_config(descriptor);self.settings=self.descriptor['settings']
        for task,path in self.descriptor['routes'].items():
            for name,expected in self.descriptor['adapter_sha256'][task].items():
                if file_hash(Path(path)/name)!=expected:raise ValueError('Routed adapter changed')
        self.codec=LosslessSemCodec.from_dict(read_config(Path(self.settings['codec_root'])/'codec.json'))
        self.tokenizer=AutoTokenizer.from_pretrained(Path(self.settings['codec_root'])/'tokenizer',local_files_only=True)
        self.model,_,_=build_model(self.settings,self.codec,Path(self.descriptor['routes']['editing']).parent)
        self.model.load_adapter(self.descriptor['routes']['generation'],adapter_name='generation',is_trainable=False)
        self.model.requires_grad_(False);self.model.eval()

    def prompt(self,instruction,source=None):
        from .instruction_data import instruction_tokens
        pairs=self.codec.encode_pairs(source) if source is not None else None
        return instruction_tokens(self.tokenizer,self.codec.boundaries,instruction,pairs)

    def predict(self,instruction,source=None):
        from .lossless_sft import generate_one
        task='generation' if source is None else 'editing'
        self.model.set_adapter('generation' if task=='generation' else 'default')
        self.model.requires_grad_(False);self.model.eval()
        prompt=self.prompt(instruction,source)
        if len(prompt)+self.settings['max_new_tokens']>self.settings['context_length']:
            raise ValueError('Routed prompt exceeds context budget')
        sequence,ids=generate_one(self.model,prompt,self.codec,self.settings)
        return {'task':task,'sequence':sequence,'generated_ids':ids}

    def predict_many(self,requests,batch_size=None):
        """Group by explicit task and prompt length; retain request order."""
        from .batch_decode import generate_many
        import torch
        import gc
        self.last_batch_stats=[]
        result=[None]*len(requests)
        for task in ('generation','editing'):
            cases=[]
            for index,item in enumerate(requests):
                selected='generation' if item.get('source') is None else 'editing'
                if selected==task:
                    prompt=self.prompt(item['instruction'],item.get('source'))
                    if len(prompt)+self.settings['max_new_tokens']>self.settings['context_length']:
                        raise ValueError('Routed prompt exceeds context budget')
                    cases.append((index,prompt))
            if not cases:continue
            cases.sort(key=lambda pair:(len(pair[1]),pair[0]))
            size=batch_size if batch_size is not None else self.descriptor.get('inference_options',{}).get('batch_sizes',{}).get(task,1)
            if size<1:raise ValueError('Batch size must be positive')
            self.model.set_adapter('generation' if task=='generation' else 'default')
            self.model.requires_grad_(False);self.model.eval();offset=0
            while offset<len(cases):
                chosen=cases[offset:offset+size];retry=False
                try:
                    if len(chosen)==1:
                        from .lossless_sft import generate_one
                        sequence,ids=generate_one(self.model,chosen[0][1],self.codec,self.settings)
                        rows=[{'sequence':sequence,'generated_ids':ids,'error':None}]
                    else:rows=generate_many(self.model,[p for _,p in chosen],self.codec,self.settings)
                except torch.OutOfMemoryError:
                    if len(chosen)==1:raise
                    self.last_batch_stats.append({'task':task,'event':'oom','batch_size':len(chosen)})
                    size=max(1,len(chosen)//2);retry=True
                if retry:
                    gc.collect();torch.cuda.empty_cache();continue
                for (index,_),row in zip(chosen,rows):result[index]={'task':task,**row}
                self.last_batch_stats.append({'task':task,'event':'batch','batch_size':len(chosen)})
                offset+=len(chosen)
        return result
