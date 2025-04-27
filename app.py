import streamlit as st
from transformers import LlamaTokenizer, LlamaForCausalLM
from huggingface_hub import login
from peft import PeftModel

login(token="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")

# Bước 1: Load base model
base_model_name = "meta-llama/Llama-3.2-1B"  # base model gốc
adapter_name = "thainq107/Llama-3.2-1B-Instruct-dpo"  # adapter model

tokenizer = LlamaTokenizer.from_pretrained(base_model_name)
model = LlamaForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,  # hoặc float32 nếu máy bạn không có GPU
    device_map="auto"
)

# Bước 2: Apply adapter
model = PeftModel.from_pretrained(model, adapter_name)

def generate_response(sentence):
    inputs = tokenizer(
        [{"role": "user", "content": sentence}],
        return_tensors="pt"
    ).to(model.device)

    output_tokens = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7
    )

    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return output_text

def main():
    st.title('Vietnamese Custom Preference Model')
    st.header('Model: LLaMA-3.2-1B. Dataset: Alpaca-Vi')
    text_input = st.text_input("Sentence: ", "Hãy giải thích về trí tuệ nhân tạo bằng tiếng Việt")
    output = generate_response(text_input)
    st.success(output)

if __name__ == '__main__':
     main() 
