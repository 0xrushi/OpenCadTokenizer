import copy
from pathlib import Path
import tempfile
import unittest
from cadtokenizer.common import model_runtime_available
if not model_runtime_available():
    raise unittest.SkipTest('Model checks require the authorized image')
import torch
from cadtokenizer.lossless_model import SplitEmbedding,SplitHead,install_rows


class LosslessModelTests(unittest.TestCase):
    def test_old_rows_are_frozen_and_only_new_rows_receive_weight_gradients(self):
        torch.manual_seed(7)
        embedding=torch.nn.Embedding(8,4)
        head=torch.nn.Linear(4,8,bias=False)
        original_embedding=embedding.weight.detach().clone()
        original_head=head.weight.detach().clone()
        split=SplitEmbedding(embedding,torch.randn(3,4))
        output=SplitHead(head,torch.randn(3,4))
        ids=torch.tensor([[1,8,9,2,10]])
        hidden=split(ids)
        self.assertTrue(torch.equal(hidden[0,0],original_embedding[1]))
        output(hidden).square().mean().backward()
        self.assertIsNone(embedding.weight.grad)
        self.assertIsNone(head.weight.grad)
        self.assertGreater(float(split.cad_rows.weight.grad.abs().sum()),0)
        self.assertGreater(float(output.cad_rows.weight.grad.abs().sum()),0)
        self.assertTrue(torch.equal(embedding.weight,original_embedding))
        self.assertTrue(torch.equal(head.weight,original_head))

    def test_peft_row_and_lora_checkpoint_restores_logits(self):
        from transformers import LlamaConfig,LlamaForCausalLM
        from peft import LoraConfig,PeftModel,get_peft_model
        def base():
            torch.manual_seed(7)
            cfg=LlamaConfig(vocab_size=32,hidden_size=32,intermediate_size=64,
                num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,
                max_position_embeddings=128,tie_word_embeddings=False)
            model=LlamaForCausalLM(cfg)
            install_rows(model,torch.randn(3,32),torch.randn(3,32))
            return model
        model=get_peft_model(base(),LoraConfig(r=2,lora_alpha=2,target_modules=['q_proj','v_proj'],
            modules_to_save=['cad_rows'],task_type='CAUSAL_LM',lora_dropout=0.))
        ids=torch.tensor([[1,32,5,33,34]])
        labels=ids.clone();labels[:,:2]=-100
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
        model(input_ids=ids,labels=labels).loss.backward()
        row_names=[name for name,p in model.named_parameters() if p.requires_grad and 'cad_rows.modules_to_save.' in name]
        self.assertEqual(len(row_names),2)
        for name,p in model.named_parameters():
            if name in row_names:
                self.assertGreater(float(p.grad.abs().sum()),0)
            if '.base.weight' in name:
                self.assertFalse(p.requires_grad)
                self.assertIsNone(p.grad)
        optimizer.step();model.eval()
        with torch.no_grad(): expected=model(input_ids=ids).logits
        with tempfile.TemporaryDirectory() as temporary:
            model.save_pretrained(temporary,save_embedding_layers=False)
            self.assertLess((Path(temporary)/'adapter_model.safetensors').stat().st_size,100000)
            restored=PeftModel.from_pretrained(base(),temporary,is_trainable=True).eval()
            with torch.no_grad(): actual=restored(input_ids=ids).logits
            self.assertTrue(torch.allclose(expected,actual,atol=1e-6,rtol=1e-6))
            generated=restored.generate(input_ids=ids,attention_mask=torch.ones_like(ids),
                max_new_tokens=2,do_sample=False,pad_token_id=0,eos_token_id=2)
            self.assertGreater(generated.shape[1],ids.shape[1])
