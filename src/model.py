"""A small, CPU-friendly text classifier.

Embedding -> masked mean pool over tokens -> two-layer head. Deliberately
simple so it trains in seconds on the sample and is easy to reason about.
Swap in an LSTM or a fine-tuned DistilBERT later without touching the rest.
"""
import torch
import torch.nn as nn


class TextClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, embed_dim: int = 64, pad_idx: int = 0):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        emb = self.embedding(x)                       # (B, L, D)
        mask = (x != self.pad_idx).unsqueeze(-1)      # (B, L, 1)
        summed = (emb * mask).sum(dim=1)              # (B, D)
        counts = mask.sum(dim=1).clamp(min=1)         # (B, 1)
        pooled = summed / counts                      # masked mean
        return self.head(pooled)
