"""Small trainable CAD vocabulary extensions to a frozen base model."""
from .common import require_container
require_container()
import torch
from torch import nn


class SplitEmbedding(nn.Module):
    def __init__(self, base, rows):
        super().__init__()
        self.base=base.requires_grad_(False)
        self.old_size=base.num_embeddings
        self.embedding_dim=base.embedding_dim
        self.num_embeddings=self.old_size+len(rows)
        self.padding_idx=base.padding_idx
        self.cad_rows=nn.Embedding.from_pretrained(rows.clone(),freeze=False)

    @property
    def weight(self):
        return torch.cat((self.base.weight,self.cad_rows.weight),dim=0)

    def forward(self,ids):
        old=self.base(ids.clamp(max=self.old_size-1))
        new=self.cad_rows((ids-self.old_size).clamp(min=0))
        return torch.where((ids>=self.old_size)[...,None],new.to(old.dtype),old)


class SplitHead(nn.Module):
    def __init__(self,base,rows):
        super().__init__()
        self.base=base.requires_grad_(False)
        self.in_features=base.in_features
        self.out_features=base.out_features+len(rows)
        self.cad_rows=nn.Linear(self.in_features,len(rows),bias=False,device=rows.device,dtype=rows.dtype)
        with torch.no_grad():
            self.cad_rows.weight.copy_(rows)

    @property
    def weight(self):
        return torch.cat((self.base.weight,self.cad_rows.weight),dim=0)

    def forward(self,hidden):
        old=self.base(hidden)
        return torch.cat((old,self.cad_rows(hidden).to(old.dtype)),dim=-1)


def install_rows(model,input_rows,output_rows):
    if model.config.tie_word_embeddings:
        raise ValueError('Pilot requires untied LLaMA vocabulary weights')
    old_size=model.get_input_embeddings().num_embeddings
    if model.get_output_embeddings().out_features!=old_size or input_rows.shape!=output_rows.shape:
        raise ValueError('Incompatible vocabulary extension')
    model.set_input_embeddings(SplitEmbedding(model.get_input_embeddings(),input_rows))
    model.set_output_embeddings(SplitHead(model.get_output_embeddings(),output_rows))
    model.config.vocab_size=old_size+len(input_rows)


def semantic_row_initialization(model,tokenizer,codec):
    from .representation import NAMES
    old_size=model.get_input_embeddings().num_embeddings
    ids=list(codec.native_ids)+list(codec.boundaries.values())
    if sorted(ids)!=list(range(old_size,old_size+len(ids))):
        raise ValueError('CAD IDs must be a contiguous vocabulary extension')
    labels={native:name.strip('<>').replace('_',' ') for native,name in zip(codec.native_ids,NAMES[1:])}
    labels.update({codec.boundaries['start']:'CAD sequence',codec.boundaries['pair_end']:'pair boundary',codec.boundaries['end']:'CAD end'})
    inputs,outputs,metadata=[],[],[]
    with torch.no_grad():
        for native in sorted(ids):
            text_ids=tokenizer.encode(labels[native],add_special_tokens=False)
            if not text_ids or any(value>=old_size for value in text_ids):
                raise ValueError('Initialization must use original vocabulary')
            indices=torch.tensor(text_ids,device=model.get_input_embeddings().weight.device)
            inputs.append(model.get_input_embeddings().weight[indices].float().mean(0).to(model.dtype))
            outputs.append(model.get_output_embeddings().weight[indices].float().mean(0).to(model.dtype))
            metadata.append({'native_id':native,'text':labels[native],'base_text_ids':text_ids})
    return torch.stack(inputs),torch.stack(outputs),metadata
