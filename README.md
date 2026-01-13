# 🧠 Synapse - A Custom Mini Language Model

> **Built from Scratch by Abhinav**  
> A complete implementation of a Transformer-based Language Model with custom BPE Tokenizer.

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Architecture Deep Dive](#-architecture-deep-dive)
3. [BPE Tokenizer Explained](#-bpe-tokenizer-explained)
4. [Training Loop Dissected](#-training-loop-dissected)
5. [Loss vs Perplexity](#-loss-vs-perplexity)
6. [How Attention Heads Work](#-how-attention-heads-work)
7. [How Prediction Works](#-how-prediction-works)
8. [Files Structure](#-files-structure)
9. [Usage](#-usage)
10. [Results & Graphs](#-results--graphs)

---

## 🌟 Overview

**Synapse** is a mini GPT-style language model built entirely from scratch to demonstrate:

- ✅ Custom **BPE (Byte Pair Encoding) Tokenizer** - No HuggingFace, no tiktoken
- ✅ **Decoder-only Transformer** architecture (like GPT-2/3/4)
- ✅ **Multi-Head Self-Attention** mechanism
- ✅ Complete **training pipeline** with logging
- ✅ **Inference/Chat** capability

### Model Specifications

| Component | Value |
|-----------|-------|
| Architecture | Decoder-only Transformer |
| Embedding Dimension | 128 |
| Attention Heads | 4 |
| Transformer Layers | 4 |
| Context Window | 128 tokens |
| Vocab Size | ~1500 tokens |
| Total Parameters | ~800K |

---

## 🏗️ Architecture Deep Dive

### High-Level Flow

```
Input Text → Tokenizer → Embeddings → Transformer Blocks → LM Head → Next Token
```

### Detailed Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT TEXT                               │
│                      "User: Hello"                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BPE TOKENIZER                               │
│  • Regex split (GPT-2 pattern)                                   │
│  • UTF-8 encoding                                                │
│  • Apply learned merges                                          │
│  Output: [256, 432, 67, ...]                                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              TOKEN EMBEDDING TABLE                               │
│  nn.Embedding(vocab_size, 128)                                   │
│  Each token ID → 128-dimensional vector                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│            POSITIONAL EMBEDDING TABLE                            │
│  nn.Embedding(128, 128)                                          │
│  Position 0,1,2,...127 → 128-dim vectors                         │
│  (Tells model WHERE each token is in the sequence)               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼ (Token Embedding + Position Embedding)
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│                   TRANSFORMER BLOCK × 4                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │            MULTI-HEAD SELF-ATTENTION                        │  │
│  │  • 4 Attention Heads (32-dim each)                          │  │
│  │  • Each head learns different "patterns"                    │  │
│  │  • Causal mask prevents seeing future                       │  │
│  └────────────────────────────────────────────────────────────┘  │
│                           │                                      │
│                           ▼ + Residual Connection                │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                    LAYER NORM                               │  │
│  └────────────────────────────────────────────────────────────┘  │
│                           │                                      │
│                           ▼                                      │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              FEED-FORWARD NETWORK                           │  │
│  │  Linear(128 → 512) → ReLU → Linear(512 → 128)               │  │
│  │  (Processes each position independently)                    │  │
│  └────────────────────────────────────────────────────────────┘  │
│                           │                                      │
│                           ▼ + Residual Connection                │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                    LAYER NORM                               │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ (Repeat 4 times)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FINAL LAYER NORM                              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LM HEAD                                     │
│  Linear(128 → vocab_size)                                        │
│  Outputs "logits" - raw scores for each possible next token      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                 SOFTMAX + SAMPLING                               │
│  logits → probabilities → sample next token                      │
│  Output: "Hi" (or token ID 847)                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔤 BPE Tokenizer Explained

### What is BPE?

**Byte Pair Encoding (BPE)** is a subword tokenization algorithm that:
1. Starts with individual bytes (256 base tokens)
2. Iteratively merges the most frequent adjacent byte pairs
3. Creates a vocabulary of common subwords

### Why BPE?

| Approach | Vocabulary | Problem |
|----------|------------|---------|
| Character-level | 256 | Sequences too long, hard to learn |
| Word-level | 50,000+ | Can't handle new/rare words |
| **BPE (Subword)** | 1,000-50,000 | Best of both worlds! |

### Our Implementation

```python
# GPT-2 Regex Pattern (used for pre-tokenization)
GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

This pattern ensures:
- Contractions stay together: `don't` → `don` + `'t`
- Numbers don't merge with letters: `AI2` → `AI` + `2`
- Spaces are preserved: ` the` (space + word)

### Training Process

```
Step 1: Start with raw bytes [72, 101, 108, 108, 111]  # "Hello"
Step 2: Count all adjacent pairs
        (108, 108) appears most → merge into token 256
Step 3: Repeat until vocab_size reached
```

### Encoding Example

```
Input:  "Hello world"
Bytes:  [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]
BPE:    [256, 432, 67, 845]  (after merges)
```

---

## 🔄 Training Loop Dissected

### The Complete Training Process

```python
for iter in range(max_iters):
    # 1. GET BATCH
    xb, yb = get_batch('train')
    # xb: Input tokens [batch, block_size]
    # yb: Target tokens [batch, block_size] (shifted by 1)
    
    # 2. FORWARD PASS
    logits, loss = model(xb, yb)
    # logits: [batch, block_size, vocab_size]
    # loss: Cross-entropy between predictions and targets
    
    # 3. BACKWARD PASS
    optimizer.zero_grad()      # Clear old gradients
    loss.backward()            # Compute gradients (backprop)
    optimizer.step()           # Update weights
```

### What is a Batch?

```
Dataset: "User: Hello\nAssistant: Hi there!"

Block 1: "User: Hello\nAssist"     → Target: "ser: Hello\nAssista"
Block 2: "ant: Hi there!"          → Target: "nt: Hi there!..."

Batch = Multiple blocks processed together (parallel)
```

### Gradient Descent Visualization

```
                    Loss Surface
                    ~~~~~~~~~~~~
                   /            \
                  /              \    ← Start (random weights)
                 /                \
                •←←←←←←←←←←←←←←←←←•
               /    Gradient       \
              /     descent path    \
             ↓                       ↓
            [Minimum: optimal weights]
```

---

## 📊 Loss vs Perplexity

### Cross-Entropy Loss

The loss function measures **how wrong** our predictions are:

```
Loss = -log(P(correct_token))
```

- If model is 100% sure of correct token: Loss = 0
- If model is 50% sure: Loss = 0.69
- If model is 1% sure: Loss = 4.6

### Perplexity

**Perplexity = e^(loss)** = "How confused is the model?"

| Perplexity | Interpretation |
|------------|----------------|
| 1 | Perfect prediction (impossible) |
| 10 | Like choosing from 10 equally likely options |
| 100 | Like choosing from 100 options |
| 1000+ | Model is very confused |

### Training Progress

```
Step 0:    Loss = 7.5,  Perplexity = 1800  (Random guessing)
Step 1000: Loss = 0.3,  Perplexity = 1.35  (Learning patterns)
Step 5000: Loss = 0.05, Perplexity = 1.05  (Near perfect on training data)
```

---

## 🧠 How Attention Heads Work

### The Core Idea

Each attention head asks: **"Which other tokens should I pay attention to?"**

### Query, Key, Value (QKV)

For each token:
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"
- **Value (V)**: "What information do I provide?"

### Attention Computation

```python
# For each head:
Q = x @ W_q  # Project input to queries
K = x @ W_k  # Project input to keys
V = x @ W_v  # Project input to values

# Attention scores
scores = Q @ K.T / sqrt(d_k)  # How much each token attends to others
scores = apply_causal_mask(scores)  # Can't look at future!
weights = softmax(scores)

# Weighted combination
output = weights @ V
```

### Causal Mask (Why it matters)

```
Without mask (BAD - cheating!):
Token 3 could see: [Token1, Token2, Token3, Token4, Token5]

With causal mask (GOOD):
Token 3 can see:  [Token1, Token2, Token3, -, -]
```

### Multi-Head Attention

4 heads = 4 different "perspectives"

```
Head 1: Might learn "subject-verb agreement"
Head 2: Might learn "recent context importance"
Head 3: Might learn "punctuation patterns"
Head 4: Might learn "semantic relationships"
```

---

## 🎯 How Prediction Works

### Step-by-Step Generation

```
Input: "User: What is your name?\nAssistant:"

Step 1: Tokenize → [256, 432, 67, 845, 32, ...]
Step 2: Forward pass → logits for position n+1
Step 3: Softmax → probability distribution
Step 4: Sample → token_id = 847 ("My")
Step 5: Append → [256, 432, 67, 845, 32, ..., 847]
Step 6: Repeat from Step 2
```

### Sampling Strategies

| Method | Description |
|--------|-------------|
| Greedy | Always pick highest probability token |
| Random | Sample according to distribution |
| Top-k | Sample from top k most likely |
| Top-p | Sample from tokens with cumulative prob > p |
| Temperature | Scale logits (higher = more random) |

Our implementation uses **random sampling** (torch.multinomial).

---

## 📁 Files Structure

```
tokeniser/
├── bpe_tokenizer.py        # Custom BPE Tokenizer implementation
├── tiny_gpt.py             # Transformer architecture (TinyGPT class)
├── train_llm.py            # Basic training script
├── train_with_logging.py   # Training with metrics logging
├── test_llm.py             # Interactive chat interface
├── prepare_data.py         # Dataset preparation (includes identity)
├── visualize_training.py   # Generate plots and diagrams
├── chat_data.txt           # Training data (conversations)
├── chat_tokenizer.json     # Saved tokenizer state
├── synapse_model.pth       # Trained model weights
├── training_log.csv        # Training metrics
└── README.md               # This file!
```

---

## 🚀 Usage

### 1. Prepare Data
```bash
python prepare_data.py
```

### 2. Train Model
```bash
python train_with_logging.py
```

### 3. Chat with Synapse
```bash
python test_llm.py
```

### 4. Visualize Training
```bash
pip install matplotlib
python visualize_training.py
```

---

## 📈 Results & Graphs

### Training Curves

After training, `visualize_training.py` generates:

1. **Loss Curve** (`loss_perplexity_curves.png`)
   - Shows train/val loss decreasing over time
   - Convergence indicates learning

2. **Perplexity Curve**
   - Shows model's "confusion" decreasing
   - Final perplexity ~1.05 = near-perfect on training data

3. **Tokenizer Analysis** (`tokenizer_analysis.png`)
   - Compression ratio: ~3-4x (fewer tokens than characters)
   - Vocabulary distribution

### Sample Chat

```
You: What is your name?
Synapse: My name is Synapse.

You: Who created you?
Synapse: I was created by Abhinav.

You: Hello
Synapse: Hi there! I am Synapse, how can I help you today?
```

---

## 🔬 Technical Notes

### Why This Architecture Works

1. **Attention is All You Need**: Self-attention allows every token to "communicate" with every previous token, enabling long-range dependencies.

2. **Residual Connections**: Add input to output of each sublayer. This helps gradients flow during backprop (solves vanishing gradient).

3. **Layer Normalization**: Stabilizes training by normalizing activations.

4. **Position Embeddings**: Transformers have no inherent notion of order; position embeddings inject this information.

### Limitations

- **Small Context**: 128 tokens (~50-100 words)
- **Limited Parameters**: ~800K (GPT-3 has 175B)
- **Memorization**: With small data, model memorizes rather than generalizes
- **No Instruction Tuning**: Trained on raw conversations, not with RLHF

### Future Improvements

- [ ] Increase model size (more layers, larger embeddings)
- [ ] Train on larger/diverse datasets
- [ ] Implement attention caching for faster inference
- [ ] Add temperature and top-p sampling
- [ ] Implement RLHF for better responses

---

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) - Inspiration for this project
- [BPE Algorithm](https://en.wikipedia.org/wiki/Byte_pair_encoding)

---

<p align="center">
  <b>Built with ❤️ by Abhinav</b><br>
  <i>Synapse - A journey from bytes to intelligence</i>
</p>
