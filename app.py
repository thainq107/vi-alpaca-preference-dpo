import streamlit as st
from transformers import pipeline

generator = pipeline("text-generation", model="thainq107/Llama-3.2-1B-Instruct-dpo", device="cpu")

def main():
    st.title('Vietnamese Custom Preference Model')
    st.header('Model: LLaMA-3.2-1B. Dataset: Alpaca-Vi')
    text_input = st.text_input("Sentence: ", "Hãy giải thích về trí tuệ nhân tạo bằng tiếng Việt")
    output = generator([{"role": "user", "content": question}], max_new_tokens=128)[0]
    st.success(output)

if __name__ == '__main__':
     main() 
