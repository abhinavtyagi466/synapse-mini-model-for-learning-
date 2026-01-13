# Synapse - Custom Mini Language Model

A from-scratch implementation of a Transformer-based Language Model with custom BPE Tokenizer.

**Built by Abhinav**

---

## Model Specifications

| Component | Value |
|-----------|-------|
| Architecture | Decoder-only Transformer |
| Embedding Dimension (`n_embd`) | 128 |
| Attention Heads (`n_head`) | 4 |
| Head Size | 32 (128 / 4) |
| Transformer Layers (`n_layer`) | 4 |
| Context Window (`block_size`) | 128 tokens |
| Vocabulary Size | 1500 tokens |
| Feed-Forward Hidden Dim | 512 (4 × 128) |
| Dropout | 0.1 |
| **Total Parameters** | **1,193,692** |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Learning Rate | 5e-4 |
| Optimizer | AdamW |
| Training Steps | 7000 |
| Eval Interval | 500 steps |
| Training Data Size | 573,408 characters |
| Device | CPU |

---

## Architecture

```
Input Text
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    BPE TOKENIZER (ProTokenizer)             │
│  • Pre-tokenization: GPT-2 Regex Pattern                    │
│  • Base vocab: 256 (raw bytes)                              │
│  • Learned merges: 1244 (total vocab: 1500)                 │
│  • Encoding: Rank-based merge application                   │
└─────────────────────────────────────────────────────────────┘
    │
    ▼ Token IDs [batch_size, block_size] = [32, 128]
    │
┌─────────────────────────────────────────────────────────────┐
│            TOKEN EMBEDDING: nn.Embedding(1500, 128)         │
│            Size: 1500 × 128 = 192,000 parameters            │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│         POSITION EMBEDDING: nn.Embedding(128, 128)          │
│         Size: 128 × 128 = 16,384 parameters                 │
└─────────────────────────────────────────────────────────────┘
    │
    ▼ x = token_emb + position_emb → [32, 128, 128]
    │
╔═════════════════════════════════════════════════════════════╗
║              TRANSFORMER BLOCK × 4                          ║
║                                                             ║
║  ┌─────────────────────────────────────────────────────┐    ║
║  │              LAYER NORM 1: nn.LayerNorm(128)        │    ║
║  │              Parameters: 256 (weight + bias)        │    ║
║  └─────────────────────────────────────────────────────┘    ║
║                          │                                  ║
║                          ▼                                  ║
║  ┌─────────────────────────────────────────────────────┐    ║
║  │           MULTI-HEAD SELF-ATTENTION                 │    ║
║  │                                                     │    ║
║  │   4 Heads × (Q, K, V projections):                  │    ║
║  │   • Key:   nn.Linear(128, 32, bias=False)           │    ║
║  │   • Query: nn.Linear(128, 32, bias=False)           │    ║
║  │   • Value: nn.Linear(128, 32, bias=False)           │    ║
║  │   Per head: 3 × (128 × 32) = 12,288 params          │    ║
║  │   4 heads: 49,152 params                            │    ║
║  │                                                     │    ║
║  │   Output projection: nn.Linear(128, 128)            │    ║
║  │   Params: 128 × 128 + 128 = 16,512                  │    ║
║  │                                                     │    ║
║  │   Attention formula:                                │    ║
║  │   scores = (Q @ K.T) / sqrt(32)                     │    ║
║  │   scores = mask_future(scores)  # causal            │    ║
║  │   weights = softmax(scores)                         │    ║
║  │   output = weights @ V                              │    ║
║  └─────────────────────────────────────────────────────┘    ║
║                          │                                  ║
║                          ▼ + Residual                       ║
║  ┌─────────────────────────────────────────────────────┐    ║
║  │              LAYER NORM 2: nn.LayerNorm(128)        │    ║
║  │              Parameters: 256                        │    ║
║  └─────────────────────────────────────────────────────┘    ║
║                          │                                  ║
║                          ▼                                  ║
║  ┌─────────────────────────────────────────────────────┐    ║
║  │              FEED-FORWARD NETWORK                   │    ║
║  │   nn.Linear(128, 512): 128×512 + 512 = 66,048       │    ║
║  │   ReLU()                                            │    ║
║  │   nn.Linear(512, 128): 512×128 + 128 = 65,664       │    ║
║  │   Total: 131,712 params                             │    ║
║  └─────────────────────────────────────────────────────┘    ║
║                          │                                  ║
║                          ▼ + Residual                       ║
║                                                             ║
║  Per Block Total: ~197,888 parameters                       ║
║  4 Blocks: ~791,552 parameters                              ║
╚═════════════════════════════════════════════════════════════╝
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│            FINAL LAYER NORM: nn.LayerNorm(128)              │
│            Parameters: 256                                  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│              LM HEAD: nn.Linear(128, 1500)                  │
│              Parameters: 128 × 1500 + 1500 = 193,500        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼ Logits [32, 128, 1500]
    │
┌─────────────────────────────────────────────────────────────┐
│              SOFTMAX → SAMPLE → NEXT TOKEN                  │
└─────────────────────────────────────────────────────────────┘
```

---

## BPE Tokenizer

### Implementation Details

| Component | Value |
|-----------|-------|
| Pre-tokenization | GPT-2 Regex Pattern |
| Base Vocabulary | 256 (UTF-8 bytes) |
| Learned Merges | 1244 |
| Final Vocabulary | 1500 |
| Special Tokens | `<\|endoftext\|>` (ID: 100000) |

### GPT-2 Regex Pattern

```python
GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

This pattern:
- Keeps contractions together (`'s`, `'t`, `'re`)
- Separates letters from numbers
- Handles whitespace as prefix to words

### Training Algorithm

```
1. Start with 256 byte tokens
2. For each merge iteration:
   a. Count all adjacent token pairs
   b. Find most frequent pair
   c. Merge pair → new token (next available ID)
   d. Update vocabulary
3. Repeat until vocab_size = 1500
```

### Encoding Process

```
Text: "Hello"
      ↓
Pre-tokenize (Regex split)
      ↓
UTF-8 bytes: [72, 101, 108, 108, 111]
      ↓
Apply merges (lowest rank first)
      ↓
Final tokens: [256, 432, ...]
```

---

## Self-Attention Mechanism

Each attention head computes:

```
Q = x @ W_q    # Query: "What am I looking for?"
K = x @ W_k    # Key: "What do I contain?"
V = x @ W_v    # Value: "What information to pass?"

Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
```

### Causal Mask

Lower triangular mask prevents attending to future tokens:

```
Position:  1  2  3  4
        ┌─────────────┐
Pos 1   │ 1  0  0  0  │  ← Can only see itself
Pos 2   │ 1  1  0  0  │  ← Can see pos 1-2
Pos 3   │ 1  1  1  0  │  ← Can see pos 1-3
Pos 4   │ 1  1  1  1  │  ← Can see all
        └─────────────┘
```

---

## Training Loop

```python
for step in range(7000):
    # 1. Sample batch
    x, y = get_batch('train')  # x: input, y: target (shifted by 1)
    
    # 2. Forward pass
    logits, loss = model(x, y)
    # logits: [32, 128, 1500]
    # loss: Cross-entropy between predictions and targets
    
    # 3. Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Loss Function

Cross-entropy loss:
```
Loss = -log(P(correct_token))
```

### Perplexity

```
Perplexity = exp(Loss)
```

| Perplexity | Meaning |
|------------|---------|
| 1.0 | Perfect prediction |
| 10 | Choosing from ~10 options |
| 1500 | Random guess over vocabulary |

---

## File Structure

```
tokeniser/
├── bpe_tokenizer.py        # BPE tokenizer implementation
├── tiny_gpt.py             # Transformer model (TinyGPT class)
├── train_llm.py            # Training script
├── train_with_logging.py   # Training with CSV logging
├── test_llm.py             # Interactive chat
├── prepare_data.py         # Dataset preparation
├── visualize_training.py   # Plot generation
├── chat_data.txt           # Training data (573KB)
├── chat_tokenizer.json     # Tokenizer state
├── tiny_llm.pth            # Model weights
└── README.md
```

---

## Usage

### Train
```bash
python train_llm.py
```

### Chat
```bash
python test_llm.py
```

### Visualize
```bash
pip install matplotlib
python visualize_training.py
```

---

## Parameter Breakdown

| Layer | Parameters |
|-------|------------|
| Token Embedding (1500 × 128) | 192,000 |
| Position Embedding (128 × 128) | 16,384 |
| Transformer Block ×4 | 791,552 |
| - LayerNorm ×2 per block | 2,048 total |
| - Attention (Q,K,V + proj) | 262,656 total |
| - FFN (128→512→128) | 526,848 total |
| Final LayerNorm | 256 |
| LM Head (128 → 1500) | 193,500 |
| **Total** | **1,193,692** |

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
