"""
# @author qumu
# @date 2023/9/19
# @module hf_inference.py
"""
import sys
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer, 
    AutoModelForCausalLM,
    StoppingCriteria,
)


MODEL_TYPES = {
    
}

TOKENIZERS = {
    
}

class EotOrPadStopping(StoppingCriteria):
    """
    Args:
        start_length (:obj:`int`):
            The number of initial tokens.
    """

    def __init__(self, stop_token_id, pad_token_id):
        self.stop_token_id = stop_token_id
        self.pad_token_id = pad_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        
        return torch.logical_or(input_ids[:,-1]== self.stop_token_id, input_ids[:,-1]==self.pad_token_id).all().item()


def load_model_tokenizer(path, model_type=None, peft_path=None, torch_dtype=torch.bfloat16, quantization=None, eos_token=None, pad_token=None, batch_size=1):
    """
        load model and tokenizer by transfromers
    """
    if model_type:
        ModelClass = MODEL_TYPES.get(model_type, AutoModelForCausalLM)
    else:
        ModelClass = AutoModelForCausalLM
    TokenizerClass = TOKENIZERS.get(model_type, AutoTokenizer)
    print(f"Tokenizer Class: {TokenizerClass}, Model Class: {ModelClass}")

    config, unused_kwargs = AutoConfig.from_pretrained(
        path,
        use_flash_attn=batch_size==1,
        use_xformers=batch_size==1,
        trust_remote_code=True,
        return_unused_kwargs=True)
    
    config_dict = config.to_dict()
    
    tokenizer = TokenizerClass.from_pretrained(path, trust_remote_code=True, use_fast=False, legacy=False)
    if eos_token:
        print("input eos_token: ", eos_token)
        try:
            tokenizer.eos_token = eos_token
            tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
        except:
            print(tokenizer.eos_token, tokenizer.eos_token_id)

    elif "eos_token_id" in config_dict:
        tokenizer.eos_token = tokenizer.convert_ids_to_tokens(config.eos_token_id)
        tokenizer.eos_token_id = config.eos_token_id
    elif "eos_token" in config_dict:
        print(config.eos_token)
        tokenizer.eos_token = config.eos_token
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(config.eos_token)
        
    if pad_token:
        print("input pad_token: ", pad_token)
        try:
            tokenizer.pad_token = pad_token
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
        except:
            print(tokenizer.pad_token, tokenizer.pad_token_id)

    elif "pad_token_id" in config_dict:
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(config.pad_token_id)
        tokenizer.pad_token_id = config.pad_token_id
    elif "pad_token" in config_dict:
        print(config.eos_token)
        tokenizer.pad_token = config.pad_token
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(config.pad_token)
    
    tokenizer.padding_side = "left"
    print(f"tokenizer's eos_token: {tokenizer.eos_token}, pad_token: {tokenizer.pad_token}")
    print(f"tokenizer's eos_token_id: {tokenizer.eos_token_id}, pad_token_id: {tokenizer.pad_token_id}")
    print(tokenizer)

    base_model = ModelClass.from_pretrained(
        path,
        config=config,
        load_in_8bit=(quantization=='8bit'),
        load_in_4bit=(quantization=='4bit'),
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        # use_safetensors=False,
    )

    print("Loading Original MODEL...")
    model = base_model

    model.eval()

    print("=======================================MODEL Configs=====================================")
    print(model.config)
    print("=========================================================================================")
    print("=======================================MODEL Archetecture================================")
    print(model)
    print("=========================================================================================")
    
    return model, tokenizer


def hf_inference(model, tokenizer, text_list, args=None, max_new_tokens=512, do_sample=True, **kwargs):
    """
        transformers models inference by huggingface
    """
    inputs = tokenizer(text_list, return_tensors='pt', padding=True, add_special_tokens=False).to("cuda")
    # print(inputs["attention_mask"])
    print("================================Prompts and Generations=============================")
    
    outputs = model.generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **kwargs
    )

    gen_text = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    for i in range(len(text_list)):
        print('=========' * 10)
        print(f'Prompt:\n{text_list[i]}')
        gen_text[i] =  gen_text[i].replace(tokenizer.pad_token, '')
        if args and args.language=='python' and args.dataset_type=='hv' and args.next_text=='':
            print("python spaces fix")
        print(f'Generation:\n{gen_text[i]}')
    sys.stdout.flush()
    return gen_text