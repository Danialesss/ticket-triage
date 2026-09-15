"""Vocabulary, encoding, and the PyTorch Dataset for ticket text."""
from collections import Counter

import torch
from torch.utils.data import Dataset

from src.text_utils import simple_tokenize

PAD, UNK = "<pad>", "<unk>"


def build_vocab(texts, min_freq: int = 1, max_size: int = 5000) -> dict:
    counter = Counter()
    for t in texts:
        counter.update(simple_tokenize(t))
    vocab = {PAD: 0, UNK: 1}
    for word, freq in counter.most_common(max_size):
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def encode(text, vocab: dict, max_len: int = 40):
    ids = [vocab.get(tok, vocab[UNK]) for tok in simple_tokenize(text)][:max_len]
    if len(ids) < max_len:
        ids += [vocab[PAD]] * (max_len - len(ids))
    return ids


class TicketDataset(Dataset):
    def __init__(self, texts, labels, vocab, label2id, max_len: int = 40):
        self.texts = list(texts)
        self.labels = list(labels)
        self.vocab = vocab
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        x = torch.tensor(encode(self.texts[i], self.vocab, self.max_len), dtype=torch.long)
        y = torch.tensor(self.label2id[self.labels[i]], dtype=torch.long)
        return x, y
