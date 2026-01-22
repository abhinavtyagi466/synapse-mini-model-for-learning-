# Synapse v2: A Custom 3.7M Parameter Transformer Architecture 🧠

**Synapse v2** is a decoder-only Transformer model built and trained from scratch to explore the fundamentals of Large Language Models (LLMs). This project focuses on the implementation of a professional-grade BPE tokenizer and a multi-layer attention-based neural network.

---

## 🏗️ Neural Architecture (The Core)

Synapse v2 is structured using a standard Transformer block architecture with the following specifications:

| Component | Technical Detail |
| :--- | :--- |
| **Model Type** | Decoder-only Transformer |
| **Total Parameters** | 3,672,832 (~3.67M) |
| **Embedding Dimension** | 128 |
| **Attention Heads** | 4 Heads (Multi-Head Self-Attention) |
| **Transformer Layers** | 12 Sequential Blocks |
| **Context Window** | 64 Tokens |
| **Vocabulary Size** | 5,000 Patterns |
| **Dropout Rate** | 0.2 (For Regularization) |

---

## 🛠️ Specialized Tokenization: Turbo BPE

The model utilizes **ProTokenizer**, a custom Byte Pair Encoding (BPE) implementation. Unlike basic character-level tokenizers, ProTokenizer uses:
- **GPT-Standard Regex:** Pre-tokenization for precise chunking.
- **Rank-Based Encoding:** Efficient recursive merging for high-density token mapping.
- **Efficiency:** Achieves a **3.2:1 compression ratio**, allowing for faster inference and larger context processing.

---

## 📈 Training Dynamics & Convergence

The model was optimized using the **AdamW optimizer** with a learning rate of `3e-4` over 5,000 steps. 

### Key Performance Metrics:
- **Final Training Loss:** 2.04
- **Final Validation Loss:** 3.26
- **Stability:** The architecture was carefully scaled to balance parameter depth with the instruction-tuned training corpus.

Training analytics like Loss curves and Perplexity distributions can be viewed in the generated `visual_*.png` files.

---

## 🧠 How It Works: The Attention Flow

Synapse v2 follows a simple yet powerful flow to generate responses:
1. **Embedding:** Raw text is converted into high-dimensional vectors via the BPE Tokenizer.
2. **Positional Encoding:** Learned vectors are added to give the model awareness of token order.
3. **Attention Blocks:** 12 layers of multi-head attention allow the model to weight different parts of the input context.
4. **Language Head:** A linear layer maps the final state back to the 5,000-token vocabulary, selecting the most probable next token.

---

**Developed with ❤️ by Abhinav Tyagi.**
**Exploration of Neural Architectures & Cognitive Modeling.**
