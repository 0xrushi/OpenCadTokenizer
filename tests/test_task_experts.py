import tempfile
import unittest
from collections import Counter
from pathlib import Path
from cadtokenizer.common import model_runtime_available
from cadtokenizer.task_experts import make_batches,lr_factor


class TaskScheduleTests(unittest.TestCase):
    def test_identical_exposure_and_update_budget(self):
        cases=[{'example_id':f'{task}-{i}','task':task} for task,n in [('generation',512),('editing',2560)] for i in range(n)]
        plan=make_batches(cases,500,8,142)
        self.assertEqual(plan,make_batches(cases,500,8,142))
        self.assertEqual(Counter(b['task'] for b in plan),{'generation':84,'editing':416})
        joint=Counter(k for b in plan for k in b['example_ids'])
        routed=Counter()
        for task in ('generation','editing'):
            routed.update(k for b in plan if b['task']==task for k in b['example_ids'])
        self.assertEqual(joint,routed)
        self.assertEqual(sum(joint.values()),4000)
        self.assertEqual(set(joint),{c['example_id'] for c in cases})
        self.assertEqual(set(joint.values()),{1,2})
        for task in ('generation','editing'):
            self.assertEqual([lr_factor(b['index'],500,25) for b in plan if b['task']==task],
                             [lr_factor(i,500,25) for i in range(500) if ('generation' if i%6==0 else 'editing')==task])

    def test_rejects_incomplete_task_pools(self):
        with self.assertRaises(ValueError):make_batches([{'example_id':'a','task':'editing'}],2,8,42)


@unittest.skipUnless(model_runtime_available(),'PyTorch routing tests require the cached image')
class ExpertRoutingTests(unittest.TestCase):
    def test_switching_restores_lora_and_new_token_rows(self):
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM
        from peft import LoraConfig,PeftModel,get_peft_model
        from cadtokenizer.lossless_model import install_rows
        def base():
            torch.manual_seed(7)
            model=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=32,intermediate_size=64,
                num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,tie_word_embeddings=False))
            install_rows(model,torch.randn(3,32),torch.randn(3,32))
            return model
        ids=torch.tensor([[1,32,5,33,34]])
        expected=[]
        with tempfile.TemporaryDirectory() as folder:
            paths=[]
            for i in range(2):
                model=get_peft_model(base(),LoraConfig(r=2,lora_alpha=2,target_modules=['q_proj','v_proj'],
                    modules_to_save=['cad_rows'],task_type='CAUSAL_LM',lora_dropout=0.))
                labels=ids.clone();labels[:,:2]=-100;labels[:,-1]=32+i
                optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.02)
                model(input_ids=ids,labels=labels,use_cache=False).loss.backward();optimizer.step();model.eval()
                with torch.no_grad():expected.append(model(input_ids=ids,use_cache=False).logits.clone())
                path=Path(folder)/str(i);model.save_pretrained(path,save_embedding_layers=False);paths.append(path)
            self.assertFalse(torch.allclose(expected[0],expected[1]))
            routed=PeftModel.from_pretrained(base(),paths[0],is_trainable=False).eval()
            routed.load_adapter(paths[1],adapter_name='generation',is_trainable=False)
            body=id(routed.get_base_model())
            for adapter,index in [('default',0),('generation',1),('default',0)]:
                routed.set_adapter(adapter);routed.requires_grad_(False);routed.eval()
                with torch.no_grad():actual=routed(input_ids=ids,use_cache=False).logits
                self.assertTrue(torch.allclose(actual,expected[index],atol=1e-6,rtol=1e-6))
                self.assertEqual(id(routed.get_base_model()),body)
