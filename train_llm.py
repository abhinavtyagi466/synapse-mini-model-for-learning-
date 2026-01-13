import torch
import os
from bpe_tokenizer import ProTokenizer
from tiny_gpt import TinyGPT

# Hyperparameters
batch_size = 32
block_size = 128 # Increased context
max_iters = 7000 # More training for identity
eval_interval = 500
learning_rate = 5e-4 # Slightly lower for stability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100
vocab_size = 1500 # More tokens for better sentence structure

# 1. Load Data
data_file = 'chat_data.txt'
if not os.path.exists(data_file):
    print(f"Error: {data_file} not found. Running prepare_data.py...")
    os.system("python prepare_data.py")

with open(data_file, 'r', encoding='utf-8') as f:
    text = f.read()

# 2. Tokenizer Setup
tokenizer = ProTokenizer(state_file="chat_tokenizer.json")
# Train tokenizer if not already trained
if not os.path.exists("chat_tokenizer.json"):
    print("Training conversational tokenizer...")
    tokenizer.train(text, target_vocab_size=vocab_size)
else:
    print("Loading existing chat tokenizer...")

# Encode the entire dataset
print("Tokenizing chat dataset...")
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9 * len(data)) 
train_data = data[:n]
val_data = data[n:]

# Data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# 3. Model Initialization
# Update vocab_size to actual tokenizer vocab size
actual_vocab_size = len(tokenizer.vocab)
print(f"Actual Vocab Size: {actual_vocab_size}")
model = TinyGPT(vocab_size=actual_vocab_size, block_size=block_size, device=device).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 4. Training Loop
print("Starting training...")
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# 5. Save the model
torch.save(model.state_dict(), 'tiny_llm.pth')
print("Model saved to tiny_llm.pth")

# 6. Test Generation
print("\n--- Generating some text ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device) # Start with token 0 (usually a byte)
generated_ids = model.generate(context, max_new_tokens=100)[0].tolist()
print(tokenizer.decode(generated_ids))
