#
# MattX - The Modern Single System Image (SSI) Cluster
# 
# Copyright (c) 2026 by Matthias Rechenburg
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# Commercial licensing options are available upon request.
#

import os
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.multiprocessing as mp

# ==========================================
# HYPERPARAMETERS (Strictly CPU for MattX!)
# ==========================================
batch_size = 16
block_size = 32
max_iters = 3000
learning_rate = 1e-3
device = 'cpu' # No GPUs allowed in this dojo! 🥋
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0

torch.manual_seed(1337)

# ==========================================
# DATA LOADING & VOCABULARY
# ==========================================
# Make sure 'input.txt' is in the same directory!
if not os.path.exists('input.txt'):
    print("🚨 ERROR: 'input.txt' not found! Please download the dataset first.")
    print("wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
    exit(1)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x, y

# ==========================================
# GPT MODEL DEFINITIONS
# ==========================================
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ==========================================
# MULTIPROCESSING WORKER (The Gang Members)
# ==========================================
def train_worker(rank, num_processes, model):

    # 🚨 FIX 1: Prevent OpenMP thread explosion!
    # Keep it flat and native. 1 worker = 1 thread.
    torch.set_num_threads(1)

    # Each worker gets its own optimizer, but updates the SHARED model weights!
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Divide the total iterations among the workers
    iters_per_worker = max_iters // num_processes
    
    print(f"🚀 [MattX Worker {rank}] Online! Training for {iters_per_worker} iterations...")
    
    for iter in range(iters_per_worker):
        xb, yb = get_batch('train')
        
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Print progress every 100 steps per worker
        if iter % 100 == 0 or iter == iters_per_worker - 1:
            print(f"   -> [Worker {rank}] Step {iter}: loss = {loss.item():.4f}")

    print(f"🏁 [MattX Worker {rank}] Finished training!")

# ==========================================
# MAIN CLI & EXECUTION
# ==========================================
if __name__ == "__main__":

    # 🚨 FIX 2: Slay the Fork Deadlock Monster!
    # Force PyTorch to use 'spawn' instead of 'fork' on Linux.
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass # Already set, ignore


    parser = argparse.ArgumentParser(description="MattX Distributed AI Test")
    parser.add_argument('--action', type=str, required=True, choices=['train', 'generate'], help="Action to perform")
    parser.add_argument('--nproc', type=int, default=4, help="Number of training processes")
    args = parser.parse_args()

    model_file = 'mattx_gpt.pth'

    if args.action == 'train':
        print(f"🧠 Initializing BigramLanguageModel on CPU...")
        model = BigramLanguageModel()
        
        # SHARE THE MEMORY! This is the magic that allows multiprocessing without torchrun.
        model.share_memory()
        
        print(f"🔥 Spawning {args.nproc} training processes...")
        processes = []
        for rank in range(args.nproc):
            p = mp.Process(target=train_worker, args=(rank, args.nproc, model), name=f"Worker-{rank}")
            p.start()
            processes.append(p)
            
        # The Mother task waits for the Gang
        for p in processes:
            p.join()
            
        print(f"💾 Training complete! Saving model to {model_file}...")
        torch.save(model.state_dict(), model_file)
        print("✅ Done! You can now run with '--action generate'")

    elif args.action == 'generate':
        if not os.path.exists(model_file):
            print(f"🚨 ERROR: {model_file} not found! Please run '--action train' first.")
            exit(1)
            
        print(f"🧠 Loading trained model from {model_file}...")
        model = BigramLanguageModel()
        model.load_state_dict(torch.load(model_file, weights_only=True))
        model.eval()
        
        print("✨ Generating text...\n" + "="*40)
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated_text = decode(model.generate(context, max_new_tokens=500)[0].tolist())
        print(generated_text)
        print("="*40)

