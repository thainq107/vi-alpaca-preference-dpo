import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

def create_bnb_config():
    """
    Create BitsAndBytes configuration for 4-bit quantization
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    return bnb_config

def create_peft_config():
    """
    Create LoRA configuration for PEFT
    """
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]
    )
    return peft_config

def load_model(model_name="thainq107/Llama-3.2-1B-Instruct-dpo", cache_dir="./cache", token=None):
    """
    Load LLM model with quantization
    """
    bnb_config = create_bnb_config()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map={"": torch.cuda.current_device()} if torch.cuda.is_available() else "auto",
        token=token,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    model.config.use_cache = False
    
    return model

def get_default_hyperparameters(for_dpo=False):
    """
    Get default hyperparameters for training
    """
    batch_size = 8 if for_dpo else 16
    
    hyperparameters = {
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": True,
        "learning_rate": 3e-5,
        "logging_steps": 200,
        "num_train_epochs": 2,
        "save_strategy": "no",
        "overwrite_output_dir": True,
        "optim": "paged_adamw_8bit",
        "warmup_steps": 200,
        "bf16": True,
    }
    
    return hyperparameters
