
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class LlamaInference:
    def __init__(self, model_name="thainq107/Llama-3.2-1B-Instruct-dpo", device=None):
        """
        Initialize the inference class for Llama model
        
        Args:
            model_name: The model name or path
            device: Device to run the model on (None for auto-detection)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        # Load model with the appropriate precision based on available resources
        if self.device == "cuda" and torch.cuda.is_available():
            # Use 4-bit quantization if on GPU
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                load_in_4bit=True,
            )
        else:
            # Use 8-bit or FP16 for other cases
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                load_in_8bit=self.device == "cuda"
            )
            
        # Create the pipeline for easy generation
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device if self.device != "cpu" else -1
        )
        
    def generate(self, prompt, max_length=512, temperature=0.7, do_sample=True, top_p=0.9):
        """
        Generate text based on a prompt
        
        Args:
            prompt: The input prompt
            max_length: Maximum generation length
            temperature: Generation temperature (higher = more creative)
            do_sample: Whether to use sampling
            top_p: Top-p sampling parameter
            
        Returns:
            Generated text
        """
        # Format prompt with system and user message
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        formatted_prompt = self.tokenizer.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Generate text
        outputs = self.pipe(
            formatted_prompt,
            max_length=max_length,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Extract generated text and clean up
        response = outputs[0]['generated_text']
        # Remove the prompt from the response
        response = response[len(formatted_prompt):]
        # Remove any AI assistant prefix if present
        response = response.replace("<assistant>", "").strip()
        
        return response
    
    def chat(self, messages):
        """
        Generate a response based on a conversation history
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Assistant's response
        """
        # Apply chat template to format messages properly
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Generate text
        outputs = self.pipe(
            formatted_prompt,
            max_length=formatted_prompt.count(' ') + 512,  # Reasonable max length
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Extract generated text and clean up
        full_response = outputs[0]['generated_text']
        # Remove the prompt from the response
        response = full_response[len(formatted_prompt):]
        # Remove any assistant prefix if present
        response = response.replace("<assistant>", "").strip()
        
        return response

# Example usage
if __name__ == "__main__":
    # Create inference object
    llm = LlamaInference("thainq107/Llama-3.2-1B-Instruct-dpo")
    
    # Simple generation
    prompt = "Hãy giải thích về trí tuệ nhân tạo bằng tiếng Việt"
    response = llm.generate(prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    
    # Chat conversation
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Bạn có thể giúp tôi viết một email chúc mừng sinh nhật không?"},
    ]
    response = llm.chat(messages)
    print("\nChat:")
    print(f"User: {messages[-1]['content']}")
    print(f"Assistant: {response}")
