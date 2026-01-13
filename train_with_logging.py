"""
Synapse LLM Training Script with Detailed Logging
==================================================
This script trains the Synapse mini-LLM and logs:
- Training Loss
- Validation Loss  
- Perplexity (exp of loss)
- Learning dynamics over time

All metrics are saved to CSV for visualization.
"""

import torch
import os
import csv
import math
from bpe_tokenizer import ProTokenizer
from tiny_gpt import TinyGPT

# ============================================
# HYPERPARAMETERS (The "Knobs" of our LLM)
# ============================================
batch_size = 32          # How many sequences to process at once
block_size = 128         # Context window (how many tokens the model "remembers")
max_iters = 7000         # Total training steps
eval_interval = 100      # How often to evaluate and log metrics
learning_rate = 5e-4     # Step size for gradient descent
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50          # Batches to average for eval loss
vocab_size = 1500        # Target vocabulary size for tokenizer

print(f"Training on: {device}")

# ============================================
# DATA LOADING
# ============================================
data_file = 'chat_data.txt'
if not os.path.exists(data_file):
    print(f"Error: {data_file} not found. Run prepare_data.py first.")
    exit()

with open(data_file, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Dataset size: {len(text):,} characters")

# ============================================
# TOKENIZER SETUP
# ============================================
tokenizer = ProTokenizer(state_file="chat_tokenizer.json")
if not os.path.exists("chat_tokenizer.json"):
    print("Training conversational tokenizer...")
    tokenizer.train(text, target_vocab_size=vocab_size)
else:
    print("Loading existing chat tokenizer...")

# Tokenize the dataset
print("Tokenizing dataset...")
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")
print(f"Compression ratio: {len(text) / len(data):.2f}x (chars per token)")

# ============================================
# BATCH GENERATION
# ============================================
def get_batch(split):
    """Get a random batch of data for training or validation."""
    data_source = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_source) - block_size, (batch_size,))
    x = torch.stack([data_source[i:i+block_size] for i in ix])
    y = torch.stack([data_source[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ============================================
# LOSS ESTIMATION
# ============================================
@torch.no_grad()
def estimate_loss():
    """Estimate loss on train and val splits."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# ============================================
# MODEL INITIALIZATION
# ============================================
actual_vocab_size = len(tokenizer.vocab)
print(f"\nActual Vocab Size: {actual_vocab_size}")
print(f"Block Size (Context): {block_size}")

model = TinyGPT(
    vocab_size=actual_vocab_size,
    n_embd=128,
    n_head=4,
    n_layer=4,
    block_size=block_size,
    dropout=0.1,
    device=device
).to(device)

# Count parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Model Parameters: {num_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ============================================
# TRAINING LOOP WITH LOGGING
# ============================================
print("\n" + "="*50)
print("STARTING TRAINING")
print("="*50)

# CSV logging
log_file = 'training_log.csv'
with open(log_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['step', 'train_loss', 'val_loss', 'train_perplexity', 'val_perplexity'])

for iter in range(max_iters):
    # Evaluate and log periodically
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        train_loss = losses['train']
        val_loss = losses['val']
        
        # Perplexity = exp(loss)
        # Lower perplexity = model is more "certain" about predictions
        train_ppl = math.exp(train_loss) if train_loss < 20 else float('inf')
        val_ppl = math.exp(val_loss) if val_loss < 20 else float('inf')
        
        print(f"Step {iter:5d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train PPL: {train_ppl:.2f} | Val PPL: {val_ppl:.2f}")
        
        # Log to CSV
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([iter, train_loss, val_loss, train_ppl, val_ppl])
    
    # Training step
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ============================================
# SAVE MODEL
# ============================================
torch.save(model.state_dict(), 'synapse_model.pth')
print(f"\nModel saved to synapse_model.pth")
print(f"Training log saved to {log_file}")

# ============================================
# TEST GENERATION
# ============================================
print("\n" + "="*50)
print("SAMPLE GENERATION")
print("="*50)

test_prompts = [
    "User: What is your name?\nAssistant:",
    "User: Who created you?\nAssistant:",
    "User: Hello\nAssistant:",
]

model.eval()
for prompt in test_prompts:
    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=30)[0].tolist()
    output = tokenizer.decode(generated)
    
    # Extract just the assistant response
    response = output[len(prompt):]
    if "User:" in response:
        response = response.split("User:")[0]
    
    print(f"\nPrompt: {prompt}")
    print(f"Response: {response.strip()}")

print("\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)
