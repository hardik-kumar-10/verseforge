# train_keyword_extractor.py
import os
import re
import json
import random
import string
import argparse
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------
# Utilities
# -----------------------
def simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    tokens = text.split()
    return tokens

def build_vocab(corpus: List[str], min_freq: int = 2, max_size: int = 50000) -> Dict[str, int]:
    freq = {}
    for line in corpus:
        for tok in simple_tokenize(line):
            freq[tok] = freq.get(tok, 0) + 1
    # sort by freq
    items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    vocab = {"<pad>":0, "<unk>":1}
    for w,c in items:
        if c < min_freq: break
        if len(vocab) >= max_size: break
        vocab[w] = len(vocab)
    return vocab

def encode(tokens: List[str], vocab: Dict[str,int]) -> List[int]:
    return [vocab.get(t, vocab["<unk>"]) for t in tokens]

def pad_seq(seq: List[int], max_len: int, pad_id: int=0) -> List[int]:
    return seq[:max_len] + [pad_id] * max(0, max_len - len(seq))

# -----------------------
# Dataset
# -----------------------
class KeywordDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[List[str]], vocab: Dict[str,int], max_len: int=256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        # build label space
        label_set = set()
        for kw_list in labels:
            for kw in kw_list:
                label_set.add(kw.lower())
        self.label_list = sorted(list(label_set))
        self.label2id = {w:i for i,w in enumerate(self.label_list)}
        self.id2label = {i:w for w,i in self.label2id.items()}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        toks = simple_tokenize(self.texts[idx])
        ids = encode(toks, self.vocab)
        x = torch.tensor(pad_seq(ids, self.max_len), dtype=torch.long)
        y = torch.zeros(len(self.label_list), dtype=torch.float)
        for kw in self.labels[idx]:
            kw = kw.lower()
            if kw in self.label2id:
                y[self.label2id[kw]] = 1.0
        return x, y

# -----------------------
# Model: BiLSTM + MaxPool + MLP for multi-label
# -----------------------
class KeywordExtractor(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_labels: int, num_layers:int=1, bidirectional:bool=True, dropout:float=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout if num_layers>1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, num_labels)
        )

    def forward(self, x):
        emb = self.embed(x)                # B,T,E
        out, _ = self.lstm(emb)            # B,T,H
        out = out.max(dim=1).values        # B,H (temporal max-pool)
        out = self.dropout(out)
        logits = self.classifier(out)      # B,C
        return logits

# -----------------------
# Auto-labeling via TF-IDF (weak supervision)
# -----------------------
def auto_label_keywords(texts: List[str], top_k:int=5) -> List[List[str]]:
    vec = TfidfVectorizer(stop_words='english', lowercase=True, token_pattern=r"(?u)\b\w+\b")
    X = vec.fit_transform(texts)
    vocab = vec.get_feature_names_out()
    labels = []
    for i in range(X.shape[0]):
        row = X.getrow(i)
        nnz = list(zip(row.indices, row.data))
        nnz.sort(key=lambda x: x[1], reverse=True)
        kws = [vocab[j] for j,_ in nnz[:top_k]]
        labels.append(kws)
    return labels

# -----------------------
# Training
# -----------------------
def train(args):
    # Load your raw lyric lines or documents
    # Expect a JSONL: {"text": "..."} per line OR adapt this to MongoDB cursor fetch
    texts = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])

    # Split
    n = len(texts)
    idx = list(range(n))
    random.shuffle(idx)
    split = int(0.9 * n)
    train_idx, val_idx = idx[:split], idx[split:]
    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]

    # Auto-label if no labels file provided
    if args.labels_jsonl is None:
        train_labels = auto_label_keywords(train_texts, top_k=args.top_k)
        val_labels = auto_label_keywords(val_texts, top_k=args.top_k)
    else:
        # Expect matching order with texts; otherwise map by ID
        train_labels, val_labels = [], []
        with open(args.labels_jsonl, "r", encoding="utf-8") as f:
            all_labels = [json.loads(l)["keywords"] for l in f]
        train_labels = [all_labels[i] for i in train_idx]
        val_labels = [all_labels[i] for i in val_idx]

    # Build vocab on train
    vocab = build_vocab(train_texts, min_freq=args.min_freq, max_size=args.max_vocab)

    # Datasets
    train_ds = KeywordDataset(train_texts, train_labels, vocab, max_len=args.max_len)
    val_ds   = KeywordDataset(val_texts,   val_labels,   vocab, max_len=args.max_len)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KeywordExtractor(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_labels=len(train_ds.label_list),
        num_layers=args.num_layers,
        bidirectional=True,
        dropout=args.dropout
    ).to(device)

    # Loss & Optim
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val = 1e9
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "keyword_labels.json"), "w") as f:
        json.dump({"labels": train_ds.label_list, "vocab": vocab}, f)

    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        print(f"Epoch {epoch}: train {train_loss:.4f}, val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "vocab": vocab,
                "labels": train_ds.label_list,
                "config": vars(args)
            }, os.path.join(args.out_dir, "keyword_extractor.pt"))
            print("Saved best model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--labels_jsonl", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="artifacts/keywords")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_vocab", type=int, default=50000)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()
    train(args)
