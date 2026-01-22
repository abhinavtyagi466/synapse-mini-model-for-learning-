import torch
from bpe_tokenizer import ProTokenizer
from train_llm import ChildishGPT

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = 'synapse_v2.pth' 
tokenizer_path = 'tokenizer_state.json'

# 1. Load Tokenizer
tokenizer = ProTokenizer(state_file=tokenizer_path)
vocab_size = len(tokenizer.vocab)
print(f"Loaded Tokenizer | Vocab: {vocab_size}")

# 2. Load Model
model = ChildishGPT().to(device) 
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Synapse v2 (Stable 3.7M) Loaded Successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

def generate_safe_response(prompt, max_tokens=100):
    context_text = f"User: {prompt}\nAssistant:"
    ids = tokenizer.encode(context_text)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    
    # Greedy Generation + Manual Termination
    generated_ids = ids[:]
    
    for _ in range(max_tokens):
        idx_cond = idx[:, -64:] # Block size match
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        
        # GREEDY: Pick the ABSOLUTE best token (reduces hallucination)
        next_id = torch.argmax(logits, dim=-1).item()
        
        generated_ids.append(next_id)
        idx = torch.cat((idx, torch.tensor([[next_id]], device=device)), dim=1)
        
        # STOP CONDITION: If model says 'User:' or hits a Newline
        text_so_far = tokenizer.decode(generated_ids)
        if text_so_far.endswith("\n") or "User:" in text_so_far[len(context_text):]:
            break
            
    full_text = tokenizer.decode(generated_ids)
    response = full_text[len(context_text):].split("User:")[0].strip()
    return response

print("\n--- Synapse v2: Stable Conversation Mode ---")
while True:
    u = input("\nAbhinav: ")
    if u.lower() in ['exit', 'quit']: break
    
    res = generate_safe_response(u)
    # If model returns empty, fallback
    if not res: res = "I'm still learning that, boss."
    
    print(f"Synapse: {res}")
