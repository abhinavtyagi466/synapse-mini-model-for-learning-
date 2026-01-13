import json
import os
import regex as re

# This is the EXACT regex used by GPT-2 and adapted by most modern LLMs (GPT-4/Llama)
# It ensures:
# 1. Contractions like 's, 't, 're are kept together or split logically.
# 2. Letters (\p{L}), Numbers (\p{N}), and Whitespace are handled separately.
# 3. Prevents different types of characters from merging (e.g., "AI100" won't merge "I" and "1").
GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class ProTokenizer:
    def __init__(self, state_file="tokenizer_state.json"):
        self.state_file = state_file
        self.compiled_pattern = re.compile(GPT2_PATTERN)
        # Initial 256 bytes
        self.vocab = {i: list(bytes([i])) for i in range(256)} 
        self.merges = {} # (int, int) -> int (Rank/NewID)
        self.special_tokens = {"<|endoftext|>": 100000} # Placeholder for MOE scaling
        self.load()

    def get_stats(self, ids_list):
        """Optimized: Count pairs across multiple pre-tokenized chunks."""
        counts = {}
        for ids in ids_list:
            for pair in zip(ids, ids[1:]):
                counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge_chunk(self, ids, pair, idx):
        """Optimized chunk merging."""
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def train(self, text, target_vocab_size, verbose=True):
        if verbose: print(f"Training Professional BPE to size {target_vocab_size}...")
        
        # Pre-tokenize: Split text into logical chunks using the Regex
        # This is CRITICAL for production LLMs.
        text_chunks = re.findall(self.compiled_pattern, text)
        ids_list = [list(chunk.encode("utf-8")) for chunk in text_chunks]
        
        current_vocab_size = len(self.vocab)
        num_merges = target_vocab_size - current_vocab_size
        
        for i in range(num_merges):
            stats = self.get_stats(ids_list)
            if not stats: break
            
            # Rank-based: The most frequent pair gets the next rank
            best_pair = max(stats, key=stats.get)
            new_id = current_vocab_size + i
            
            if verbose: print(f"Rank {new_id}: Merging {best_pair} (freq: {stats[best_pair]})")
            
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
            # Update all chunks
            ids_list = [self.merge_chunk(ids, best_pair, new_id) for ids in ids_list]
        
        self.save()

    def encode(self, text):
        """Rank-based encoding (Production standard)."""
        # 1. Regex split
        text_chunks = re.findall(self.compiled_pattern, text)
        all_tokens = []
        
        for chunk in text_chunks:
            chunk_ids = list(chunk.encode("utf-8"))
            while len(chunk_ids) >= 2:
                # Find all possible pairs in current chunk
                stats = {}
                for pair in zip(chunk_ids, chunk_ids[1:]):
                    if pair in self.merges:
                        # Use the training rank to decide what to merge first
                        stats[pair] = self.merges[pair]
                
                if not stats: break # No more learned merges possible
                
                # Production Logic: Always merge the pair with the LOWEST rank (first learned)
                best_pair = min(stats, key=stats.get)
                chunk_ids = self.merge_chunk(chunk_ids, best_pair, self.merges[best_pair])
            
            all_tokens.extend(chunk_ids)
        return all_tokens

    def decode(self, ids):
        byte_list = []
        for idx in ids:
            if idx in self.vocab:
                byte_list.extend(self.vocab[idx])
        return bytes(byte_list).decode("utf-8", errors="replace")

    def save(self):
        # Technical State
        serializable_merges = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        data = {
            "merges": serializable_merges,
            "vocab": {str(k): v for k, v in self.vocab.items()},
            "special_tokens": self.special_tokens
        }
        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=4)

        # Human-Readable vocab.json (Production format)
        readable_vocab = {}
        # Add special tokens first
        for st, s_id in self.special_tokens.items():
            readable_vocab[st] = s_id
            
        for idx, byte_list in self.vocab.items():
            try:
                s = bytes(byte_list).decode('utf-8')
                s = s.replace(' ', 'Ġ').replace('\n', 'Ċ')
                readable_vocab[s] = int(idx)
            except:
                readable_vocab[f"<0x{byte_list[0]:02x}>"] = int(idx)
        
        with open("vocab.json", "w", encoding="utf-8") as f:
            json.dump(readable_vocab, f, indent=4, ensure_ascii=False)

    def load(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as f:
                data = json.load(f)
                self.merges = {tuple(map(int, k.split(","))): v for k, v in data["merges"].items()}
                self.vocab = {int(k): v for k, v in data["vocab"].items()}
                self.special_tokens = data.get("special_tokens", {})
