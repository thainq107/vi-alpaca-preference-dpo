import torch
from datasets import load_dataset
from transformers import AutoTokenizer

def load_data(model_name="thainq107/Llama-3.2-1B-Instruct-dpo", cache_dir="./cache"):
    """
    Load the dataset for fine-tuning
    """
    dataset = load_dataset("thainq107/Vi-Alpaca-Preference")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    return dataset, tokenizer

def formatting_prompt_with_chat_template(example, tokenizer):
    """
    Format examples with chat template for SFT
    """
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["chosen"]},
    ]
    prompt = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False
    )
    return prompt

def convert_to_conversational_preference_format(example):
    """
    Convert examples to DPO format
    """
    return {
        "id": example["id"],
        "prompt": [{"role": "system", 
                  "content": "You are a helpful assistant."}, 
                 {"role": "user", 
                  "content": example["question"]}],
        "chosen": [{"role": "assistant", 
                  "content": example["chosen"]}],
        "rejected": [{"role": "assistant", 
                    "content": example["rejected"]}],
    }

def prepare_dpo_dataset(dataset):
    """
    Prepare dataset for DPO training
    """
    dpo_dataset = dataset.map(convert_to_conversational_preference_format)
    return dpo_dataset
