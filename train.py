import gc
import torch
import wandb
import argparse
import warnings
from transformers import TrainingArguments
from trl import DPOTrainer, DPOConfig, SFTConfig, SFTTrainer

from data import load_data, formatting_prompt_with_chat_template, prepare_dpo_dataset
from model import load_model, create_peft_config, get_default_hyperparameters

warnings.filterwarnings("ignore")

def train_sft(model_name, cache_dir, output_dir, token=None):
    """
    Train model using SFT
    """
    # Load dataset and tokenizer
    dataset, tokenizer = load_data(model_name, cache_dir)
    
    # Load model
    base_model = load_model(model_name, cache_dir, token)
    
    # Get PEFT config
    peft_config = create_peft_config()
    
    # Get hyperparameters
    hyperparameters = get_default_hyperparameters()
    
    # Initialize wandb
    wandb.init(
        project="vi-alpaca-preference",
        name=f"{model_name.split('/')[-1]}-4bit-sft"
    )
    
    # Setup SFT trainer
    MAX_LENGTH = 512
    sft_config = SFTConfig(
        **{**hyperparameters, "output_dir": output_dir, "max_seq_length": MAX_LENGTH}
    )
    
    # Create SFT trainer
    sft_trainer = SFTTrainer(
        model=base_model,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=dataset['train'],
        formatting_func=lambda x: formatting_prompt_with_chat_template(x, tokenizer)
    )
    
    # Train model
    sft_trainer.train()
    
    # Save model to hub (optional)
    if token:
        sft_trainer.push_to_hub(output_dir, token=token)
    else:
        # Save locally
        sft_trainer.save_model(output_dir)
    
    # Free memory
    del base_model, sft_trainer
    gc.collect()
    torch.cuda.empty_cache()
    
    return output_dir

def train_dpo(model_name, cache_dir, output_dir, token=None):
    """
    Train model using DPO
    """
    # Load dataset and tokenizer
    dataset, tokenizer = load_data(model_name, cache_dir)
    
    # Convert dataset to DPO format
    dpo_dataset = prepare_dpo_dataset(dataset)
    
    # Load model
    base_model = load_model(model_name, cache_dir, token)
    
    # Get PEFT config
    peft_config = create_peft_config()
    
    # Get hyperparameters for DPO
    hyperparameters = get_default_hyperparameters(for_dpo=True)
    
    # Initialize wandb
    wandb.init(
        project="vi-alpaca-preference",
        name=f"{model_name.split('/')[-1]}-4bit-dpo"
    )
    
    # Setup DPO trainer
    MAX_LENGTH = 512
    dpo_args = DPOConfig(
        **{**hyperparameters, "output_dir": output_dir, "max_length": MAX_LENGTH}
    )
    
    # Create DPO trainer
    dpo_trainer = DPOTrainer(
        base_model,
        args=dpo_args,
        train_dataset=dpo_dataset['train'],
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    # Train model
    dpo_trainer.train()
    
    # Save model to hub (optional)
    if token:
        dpo_trainer.push_to_hub(output_dir, token=token)
    else:
        # Save locally
        dpo_trainer.save_model(output_dir)
    
    # Free memory
    del base_model, dpo_trainer
    gc.collect()
    torch.cuda.empty_cache()
    
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model using SFT or DPO")
    parser.add_argument("--method", type=str, choices=["sft", "dpo"], default="sft",
                        help="Training method: 'sft' for SFT or 'dpo' for DPO")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Base model name for SFT or SFT model name for DPO")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for the trained model")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Cache directory for model and tokenizer")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace token for model access and upload")
    
    args = parser.parse_args()
    
    # Set default output directory if not provided
    if args.output_dir is None:
        if args.method == "sft":
            args.output_dir = f"{args.model_name.split('/')[-1]}-sft"
        else:
            args.output_dir = f"{args.model_name.split('/')[-1]}-dpo"
    
    if args.method == "sft":
        train_sft(args.model_name, args.cache_dir, args.output_dir, args.token)
    else:
        train_dpo(args.model_name, args.cache_dir, args.output_dir, args.token)
