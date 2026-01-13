"""
Synapse LLM - Training Visualization
=====================================
This script reads training logs and creates publication-quality plots for:
1. Loss Curves (Train vs Validation)
2. Perplexity Curves
3. Tokenizer Compression Analysis
4. Architecture Diagram (ASCII representation)
"""

import os
import csv
import math

# Check if matplotlib is available
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not installed. Install with: pip install matplotlib")

def load_training_log(filepath='training_log.csv'):
    """Load training metrics from CSV log."""
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found. Run train_with_logging.py first.")
        return None
    
    data = {'step': [], 'train_loss': [], 'val_loss': [], 'train_ppl': [], 'val_ppl': []}
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['step'].append(int(row['step']))
            data['train_loss'].append(float(row['train_loss']))
            data['val_loss'].append(float(row['val_loss']))
            data['train_ppl'].append(float(row['train_perplexity']))
            data['val_ppl'].append(float(row['val_perplexity']))
    
    return data

def plot_loss_curves(data):
    """Plot training and validation loss over time."""
    if not HAS_MATPLOTLIB:
        return
    
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(data['step'], data['train_loss'], 'b-', label='Train Loss', linewidth=2)
    plt.plot(data['step'], data['val_loss'], 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.title('Synapse LLM: Loss Curve', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Perplexity Plot
    plt.subplot(1, 2, 2)
    # Cap perplexity at 1000 for visualization
    train_ppl_capped = [min(p, 1000) for p in data['train_ppl']]
    val_ppl_capped = [min(p, 1000) for p in data['val_ppl']]
    
    plt.plot(data['step'], train_ppl_capped, 'b-', label='Train PPL', linewidth=2)
    plt.plot(data['step'], val_ppl_capped, 'r-', label='Val PPL', linewidth=2)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Perplexity (lower = better)', fontsize=12)
    plt.title('Synapse LLM: Perplexity Curve', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('loss_perplexity_curves.png', dpi=150, bbox_inches='tight')
    print("Saved: loss_perplexity_curves.png")
    plt.show()

def plot_tokenizer_analysis():
    """Analyze and plot tokenizer compression statistics."""
    if not HAS_MATPLOTLIB:
        return
        
    from bpe_tokenizer import ProTokenizer
    
    # Load tokenizer
    tokenizer = ProTokenizer(state_file="chat_tokenizer.json")
    
    # Load sample text
    with open('chat_data.txt', 'r', encoding='utf-8') as f:
        text = f.read()[:50000]  # Sample
    
    # Character-level baseline
    char_tokens = len(text)
    
    # BPE tokens
    bpe_tokens = len(tokenizer.encode(text))
    
    # Calculate compression
    compression_ratio = char_tokens / bpe_tokens
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart: Token counts
    ax1 = axes[0]
    bars = ax1.bar(['Character-level', 'BPE Tokenizer'], [char_tokens, bpe_tokens], 
                   color=['#ff6b6b', '#4ecdc4'], edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Number of Tokens', fontsize=12)
    ax1.set_title('Tokenization: Character vs BPE', fontsize=14, fontweight='bold')
    ax1.bar_label(bars, fmt='%d', fontsize=10)
    
    # Compression ratio annotation
    ax1.annotate(f'Compression: {compression_ratio:.2f}x', 
                 xy=(1, bpe_tokens), xytext=(1.2, bpe_tokens * 2),
                 fontsize=11, ha='center',
                 arrowprops=dict(arrowstyle='->', color='green'))
    
    # Vocab distribution
    ax2 = axes[1]
    vocab = tokenizer.vocab
    token_lengths = [len(bytes(v)) for v in vocab.values()]
    
    # Histogram of token lengths
    ax2.hist(token_lengths, bins=range(1, max(token_lengths)+2), 
             color='#667eea', edgecolor='black', alpha=0.8)
    ax2.set_xlabel('Token Length (bytes)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'Vocabulary Distribution (Size: {len(vocab)})', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('tokenizer_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved: tokenizer_analysis.png")
    plt.show()

def create_architecture_diagram():
    """Create an ASCII architecture diagram."""
    diagram = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        SYNAPSE LLM ARCHITECTURE                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║   ┌─────────────────┐                                                         ║
║   │   Input Text    │  "User: Hello"                                          ║
║   └────────┬────────┘                                                         ║
║            │                                                                  ║
║            ▼                                                                  ║
║   ┌─────────────────────────────────────────────────────────────┐             ║
║   │                    BPE TOKENIZER                            │             ║
║   │  ┌─────────────────────────────────────────────────────┐    │             ║
║   │  │ 1. Split by Regex (GPT-2 Pattern)                   │    │             ║
║   │  │ 2. Convert to UTF-8 bytes                           │    │             ║
║   │  │ 3. Apply learned merges (ranked by frequency)       │    │             ║
║   │  │ 4. Output: [token_id_1, token_id_2, ...]            │    │             ║
║   │  └─────────────────────────────────────────────────────┘    │             ║
║   └────────┬────────────────────────────────────────────────────┘             ║
║            │ [256, 432, 67, 845, ...]                                         ║
║            ▼                                                                  ║
║   ┌─────────────────────────────────────────────────────────────┐             ║
║   │                  TOKEN EMBEDDING (n_embd=128)               │             ║
║   │     token_ids → learned 128-dim vectors                     │             ║
║   └────────┬────────────────────────────────────────────────────┘             ║
║            │                                                                  ║
║            ▼                                                                  ║
║   ┌─────────────────────────────────────────────────────────────┐             ║
║   │              POSITIONAL EMBEDDING (block_size=128)          │             ║
║   │     Add position info: pos(0), pos(1), ... pos(127)         │             ║
║   └────────┬────────────────────────────────────────────────────┘             ║
║            │                                                                  ║
║            ▼                                                                  ║
║   ╔═══════════════════════════════════════════════════════════════╗           ║
║   ║              TRANSFORMER BLOCK × 4 (n_layer=4)                ║           ║
║   ║ ┌───────────────────────────────────────────────────────────┐ ║           ║
║   ║ │              MULTI-HEAD SELF-ATTENTION                    │ ║           ║
║   ║ │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │ ║           ║
║   ║ │  │ Head 1  │ │ Head 2  │ │ Head 3  │ │ Head 4  │          │ ║           ║
║   ║ │  │ (32-dim)│ │ (32-dim)│ │ (32-dim)│ │ (32-dim)│          │ ║           ║
║   ║ │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘          │ ║           ║
║   ║ │       │           │           │           │               │ ║           ║
║   ║ │       └───────────┴─────┬─────┴───────────┘               │ ║           ║
║   ║ │                         │ Concatenate → 128-dim           │ ║           ║
║   ║ │                         ▼                                 │ ║           ║
║   ║ │                   ┌──────────┐                            │ ║           ║
║   ║ │                   │ Proj + LN│                            │ ║           ║
║   ║ │                   └────┬─────┘                            │ ║           ║
║   ║ └────────────────────────┼──────────────────────────────────┘ ║           ║
║   ║                          │ + Residual Connection             ║           ║
║   ║                          ▼                                   ║           ║
║   ║ ┌───────────────────────────────────────────────────────────┐ ║           ║
║   ║ │                    FEED-FORWARD MLP                       │ ║           ║
║   ║ │     Linear(128 → 512) → ReLU → Linear(512 → 128)          │ ║           ║
║   ║ └────────────────────────┬──────────────────────────────────┘ ║           ║
║   ║                          │ + Residual Connection             ║           ║
║   ╚══════════════════════════╪═══════════════════════════════════╝           ║
║                              │                                                ║
║            ▼ (Repeat 4×)     ▼                                                ║
║   ┌─────────────────────────────────────────────────────────────┐             ║
║   │                    FINAL LAYER NORM                         │             ║
║   └────────┬────────────────────────────────────────────────────┘             ║
║            │                                                                  ║
║            ▼                                                                  ║
║   ┌─────────────────────────────────────────────────────────────┐             ║
║   │              LM HEAD (Linear: 128 → vocab_size)             │             ║
║   │     Outputs probability distribution over all tokens        │             ║
║   └────────┬────────────────────────────────────────────────────┘             ║
║            │                                                                  ║
║            ▼                                                                  ║
║   ┌─────────────────┐                                                         ║
║   │ Softmax → Sample│  → Next Token → Decode → "Assistant: Hi!"               ║
║   └─────────────────┘                                                         ║
║                                                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║                        SELF-ATTENTION MECHANISM                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║   For each token position, attention computes:                               ║
║                                                                               ║
║   Query (Q): "What am I looking for?"                                        ║
║   Key (K):   "What do I contain?"                                            ║
║   Value (V): "What information do I have?"                                   ║
║                                                                               ║
║   Attention Scores = softmax(Q @ K.T / sqrt(d_k))                            ║
║   Output = Attention_Scores @ V                                              ║
║                                                                               ║
║   CAUSAL MASK (Lower Triangular):                                            ║
║   ┌─────────────────────────┐                                                ║
║   │ Tok1  Tok2  Tok3  Tok4  │                                                ║
║   │ ───────────────────────  │                                                ║
║   │ [1    0     0     0   ]  │ Tok1 can only see itself                       ║
║   │ [1    1     0     0   ]  │ Tok2 can see Tok1 and itself                   ║
║   │ [1    1     1     0   ]  │ Tok3 can see Tok1, Tok2, and itself            ║
║   │ [1    1     1     1   ]  │ Tok4 can see all previous tokens               ║
║   └─────────────────────────┘                                                ║
║                                                                               ║
║   This prevents "cheating" - model can't look at future tokens!              ║
║                                                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(diagram)
    
    # Also save to file
    with open('architecture_diagram.txt', 'w', encoding='utf-8') as f:
        f.write(diagram)
    print("Saved: architecture_diagram.txt")

def main():
    print("="*60)
    print("SYNAPSE LLM - TRAINING VISUALIZATION")
    print("="*60)
    
    # Load training data
    data = load_training_log()
    
    if data and HAS_MATPLOTLIB:
        # Plot loss curves
        print("\n1. Generating Loss & Perplexity Curves...")
        plot_loss_curves(data)
        
        # Tokenizer analysis
        print("\n2. Generating Tokenizer Analysis...")
        try:
            plot_tokenizer_analysis()
        except Exception as e:
            print(f"Could not generate tokenizer analysis: {e}")
    
    # Architecture diagram (always works)
    print("\n3. Generating Architecture Diagram...")
    create_architecture_diagram()
    
    # Summary statistics
    if data:
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Total Steps: {data['step'][-1]}")
        print(f"Final Train Loss: {data['train_loss'][-1]:.4f}")
        print(f"Final Val Loss: {data['val_loss'][-1]:.4f}")
        print(f"Final Train Perplexity: {data['train_ppl'][-1]:.2f}")
        print(f"Final Val Perplexity: {data['val_ppl'][-1]:.2f}")
        print(f"Best Val Loss: {min(data['val_loss']):.4f} at step {data['step'][data['val_loss'].index(min(data['val_loss']))]}")

if __name__ == "__main__":
    main()
