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
        if verbose: print(f"Training Professional BPE (Turbo Mode) to size {target_vocab_size}...")
        
        # 1. Pre-tokenize
        text_chunks = re.findall(self.compiled_pattern, text)
        all_ids = []
        for chunk in text_chunks:
            # We use a special marker -1 to prevent merging across chunks
            all_ids.extend(list(chunk.encode("utf-8")))
            all_ids.append(-1)
        
        if not all_ids: return

        # 2. Setup Doubly Linked List
        # pointers[i] = [prev_idx, next_idx, value]
        n = len(all_ids)
        prev_ptrs = list(range(-1, n-1))
        next_ptrs = list(range(1, n)) + [-1]
        values = all_ids
        
        # 3. Initialize Stats (Pair -> List of positions)
        # Position 'i' means the pair starts at index 'i' in the 'values' list
        pair_pos = {}
        def add_pair(i):
            v1 = values[i]
            nxt = next_ptrs[i]
            if nxt != -1:
                v2 = values[nxt]
                if v1 != -1 and v2 != -1: # Don't merge across chunk boundaries
                    p = (v1, v2)
                    if p not in pair_pos: pair_pos[p] = set()
                    pair_pos[p].add(i)

        for i in range(n-1):
            add_pair(i)

        current_vocab_size = 256
        num_merges = target_vocab_size - current_vocab_size
        
        for i in range(num_merges):
            if not pair_pos: break
            
            # Find the most frequent pair
            # We need to filter out pairs that have count 0
            # To be efficient, we'll find top pair by count (size of set)
            best_pair = max(pair_pos, key=lambda p: len(pair_pos[p]))
            if len(pair_pos[best_pair]) == 0:
                del pair_pos[best_pair]
                continue
            
            new_id = len(self.vocab)
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
            if verbose and i % 500 == 0:
                print(f"Merge {i}/{num_merges}: {best_pair} -> {new_id} (count {len(pair_pos[best_pair])})")

            # Perform Merges and update neighbors
            positions = list(pair_pos[best_pair])
            # Sort positions to handle overlapping merges like AAA -> AA correctly
            # Actually, the set.pop/remove already handles it if we are careful
            # But sorting helps ensure we don't try to merge a token that was already merged
            positions.sort()
            
            for pos in positions:
                # check if this position is still valid (not already merged by neighbor)
                if pos not in pair_pos.get(best_pair, set()): continue
                
                nxt = next_ptrs[pos]
                if nxt == -1: continue
                # The tokens being merged are values[pos] and values[nxt]
                
                # Neighbors: L -> [pos] -> [nxt] -> R
                L = prev_ptrs[pos]
                R = next_ptrs[nxt]
                
                # 1. Remove old pairs involving L, pos, nxt, R
                # Pair (L, pos)
                if L != -1:
                    p_L = (values[L], values[pos])
                    if p_L in pair_pos: 
                        pair_pos[p_L].discard(L)
                # Pair (pos, nxt) -> this is best_pair
                pair_pos[best_pair].discard(pos)
                # Pair (nxt, R)
                if R != -1:
                    p_R = (values[nxt], values[R])
                    if p_R in pair_pos:
                        pair_pos[p_R].discard(nxt)
                
                # 2. Perform merge
                values[pos] = new_id
                next_ptrs[pos] = R
                if R != -1:
                    prev_ptrs[R] = pos
                # nxt is now effectively deleted
                
                # 3. Add new pairs (L, pos) and (pos, R)
                add_pair(pos) # handles (pos, R)
                if L != -1:
                    add_pair(L) # handles (L, pos)
            
            # Cleanup best_pair if empty
            if best_pair in pair_pos and not pair_pos[best_pair]:
                del pair_pos[best_pair]
            
            current_vocab_size += 1

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
