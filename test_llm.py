import torch
from bpe_tokenizer import ProTokenizer
from tiny_gpt import TinyGPT

# Config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = 'tiny_llm.pth'
tokenizer_path = 'chat_tokenizer.json'

# 1. Load Tokenizer
tokenizer = ProTokenizer(state_file=tokenizer_path)
actual_vocab_size = len(tokenizer.vocab)
print(f"Loaded Conversational Tokenizer with Vocab Size: {actual_vocab_size}")

# 2. Load Model
actual_vocab_size = len(tokenizer.vocab)
model = TinyGPT(vocab_size=actual_vocab_size, block_size=128, device=device).to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()
print("Model Loaded Successfully!")

def generate_text(prompt, max_tokens=100):
    # Format input as a chat
    chat_prompt = f"User: {prompt}\nAssistant:"
    
    # Encode user prompt
    context_tokens = tokenizer.encode(chat_prompt)
    x = torch.tensor([context_tokens], dtype=torch.long, device=device)
    
    # Generate
    generated_indices = model.generate(x, max_new_tokens=max_tokens)[0].tolist()
    
    # Decode
    full_text = tokenizer.decode(generated_indices)
    
    # Post-process: Stop at 'User:' to avoid generating the next turn
    # We only want the Assistant's response
    response = full_text[len(chat_prompt):]
    if "User:" in response:
        response = response.split("User:")[0]
    
    return response.strip()

# Interactive Test
print("\n--- LLM Testing Mode ---")
print("Prompt dalo aur LLM usey complete karega (exit likho band karne ke liye)")

while True:
    user_input = input("\nYour Prompt: ")
    if user_input.lower() == 'exit':
        break
    
    prediction = generate_text(user_input, max_tokens=50)
    print("-" * 30)
    print(prediction)
    print("-" * 30)
