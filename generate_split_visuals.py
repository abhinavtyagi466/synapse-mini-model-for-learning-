import matplotlib.pyplot as plt
import numpy as np
import math

# --- Shared Data ---
steps = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
train_loss = [8.54, 4.22, 3.53, 3.23, 2.93, 2.73, 2.59, 2.45, 2.32, 2.16, 2.04]
val_loss = [8.53, 5.55, 5.09, 4.72, 4.43, 4.20, 3.99, 3.78, 3.62, 3.43, 3.26]
train_perplexity = [math.exp(l) for l in train_loss]
val_perplexity = [math.exp(l) for l in val_loss]

plt.style.use('dark_background')

# 1. CROSS-ENTROPY LOSS GRAPH
plt.figure(figsize=(10, 6))
plt.plot(steps, train_loss, label='Training Loss', color='#00d2ff', linewidth=2.5, marker='o', markersize=4)
plt.plot(steps, val_loss, label='Validation Loss', color='#ff007f', linewidth=2.5, marker='s', markersize=4)
plt.title('Synapse v2: Cross Entropy Loss Convergence', fontsize=14, color='#00ff88', pad=20)
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(color='#444444', linestyle='--', alpha=0.4)
plt.legend(frameon=True, facecolor='#222222')
plt.savefig('visual_1_loss.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. PERPLEXITY GRAPH
plt.figure(figsize=(10, 6))
plt.plot(steps, train_perplexity, label='Train Perplexity', color='#00ff88', linewidth=2.5)
plt.plot(steps, val_perplexity, label='Val Perplexity', color='#f8ff00', linewidth=2.5)
plt.yscale('log')
plt.title('Synapse v2: Model Perplexity (Log Scale)', fontsize=14, color='#00ff88', pad=20)
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Perplexity', fontsize=12)
plt.grid(color='#444444', linestyle='--', alpha=0.4)
plt.legend(frameon=True, facecolor='#222222')
plt.savefig('visual_2_perplexity.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. PARAMETER DISTRIBUTION
labels = ['Embeddings', 'Attention Mechanisms', 'Feed-Forward Networks', 'Language Head']
sizes = [162816, 786432, 1572864, 154624]
colors = ['#00d2ff', '#ff007f', '#00ff88', '#f8ff00']
plt.figure(figsize=(10, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140, 
        explode=(0.05, 0.05, 0.05, 0.05), textprops={'color':"w", 'fontsize':12})
plt.title('Synapse v2: Architecture Parameter Distribution', fontsize=14, color='#00ff88', pad=20)
plt.savefig('visual_3_params.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. TOKENIZER EFFICIENCY
labels_comp = ['Raw UTF-8 Bytes', 'Synapse BPE Tokens']
values_comp = [100, 31]
plt.figure(figsize=(10, 6))
bars = plt.bar(labels_comp, values_comp, color=['#444444', '#00d2ff'], width=0.6)
plt.title('Synapse v2: BPE Tokenizer Efficiency', fontsize=14, color='#00ff88', pad=20)
plt.ylabel('Sequence Length / Size', fontsize=12)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval}", ha='center', va='bottom', color='white', fontsize=12, fontweight='bold')
plt.savefig('visual_4_tokenizer.png', dpi=300, bbox_inches='tight')
plt.close()

print("Individial 4 Professional Visuals generated successfully!")
