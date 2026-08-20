from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import re


class model_runner:

    def __init__(self, model_name, cache_dir="./pretrained-models", device='cuda:0', fine_tuned_path=None):
        
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.tokenizer.padding_side = 'right'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
                
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            device_map=device,
            torch_dtype=torch.bfloat16
        )

        if fine_tuned_path is not None:
            self.model = PeftModel.from_pretrained(
                self.model,
                fine_tuned_path,
                cache_dir=self.cache_dir,
                # offload_folder="./offload2/",
                torch_dtype=torch.bfloat16,
                # offload_buffers=True
                is_trainable=False
            )
            self.model = self.model.merge_and_unload()
            # self.model.add_weighted_adapter(["default"], [1.0], "scaled_lora") 
            # self.model.set_adapter("scaled_lora")
        
        self.model.eval()
    
    def generate_from_one_prompt(self, prompt, max_length=1024, temperature=0.7, **kwargs):
        inputs = self.tokenizer(
            prompt, 
            return_tensors='pt',
            # padding=True,
            truncation=True,
            max_length=4096
        )
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        terminators = [
            self.tokenizer.eos_token_id
        ]
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id is not None and eot_id != self.tokenizer.unk_token_id:
            terminators.append(eot_id)

        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id != self.tokenizer.unk_token_id:
            terminators.append(im_end_id)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                min_new_tokens=50,
                do_sample=False,
                temperature=temperature,
                eos_token_id=terminators,
                use_cache=True,
                # repetition_penalty=1.1,
                top_p=0.95,
                **kwargs
            )
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = generated_ids[0][input_length:]
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return answer.strip()
    
    def generate_from_history(self, messages, max_length=1024, temperature=0.7, **kwargs):
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        return self.generate_from_one_prompt(prompt, max_length=max_length, temperature=temperature, **kwargs)
    
    def generate_batch(self, prompts: list, max_length=1024, temperature=0.7, **kwargs):
        self.tokenizer.padding_side = 'left' 
        inputs = self.tokenizer(
            prompts, 
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=4096
        )
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        terminators = [self.tokenizer.eos_token_id]
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id is not None and eot_id != self.tokenizer.unk_token_id:
            terminators.append(eot_id)

        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id != self.tokenizer.unk_token_id:
            terminators.append(im_end_id)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                min_new_tokens=50,
                do_sample=False,
                eos_token_id=terminators,
                use_cache=True,
                **kwargs
            )
            
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = generated_ids[:, input_length:]

        answers = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        del inputs
        del generated_ids
        del generated_tokens
        return [ans.strip() for ans in answers]
