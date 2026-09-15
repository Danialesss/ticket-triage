"""Stage 2 (PyTorch): train the classifier and save artifacts.

Run from the repo root after the Spark prep step:
    python -m src.train --train data/processed/train --test data/processed/test

If you have not run Spark yet, you can point --train/--test at the raw CSV to
smoke-test the model (it will just use the same file for both).
"""
import argparse
import json
import os

import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from src.dataset import TicketDataset, build_vocab
from src.model import TextClassifier


def load_split(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)


def main(args) -> None:
    train_df = load_split(args.train)
    test_df = load_split(args.test)
    text_col = "clean_text" if "clean_text" in train_df.columns else "text"

    labels = sorted(train_df["category"].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    vocab = build_vocab(train_df[text_col].tolist())

    train_dl = DataLoader(
        TicketDataset(train_df[text_col], train_df["category"], vocab, label2id),
        batch_size=16, shuffle=True,
    )
    test_dl = DataLoader(
        TicketDataset(test_df[text_col], test_df["category"], vocab, label2id),
        batch_size=32,
    )

    model = TextClassifier(len(vocab), len(labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for x, y in train_dl:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f"epoch {epoch + 1}/{args.epochs}  loss {running / len(train_dl):.4f}")

    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for x, y in test_dl:
            preds.extend(model(x).argmax(1).tolist())
            golds.extend(y.tolist())

    print()
    print(classification_report(
        golds, preds,
        labels=list(range(len(labels))),
        target_names=labels,
        zero_division=0,
    ))

    os.makedirs(args.out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
    with open(os.path.join(args.out, "vocab.json"), "w") as f:
        json.dump(vocab, f)
    with open(os.path.join(args.out, "labels.json"), "w") as f:
        json.dump(id2label, f)
    print(f"saved artifacts to {args.out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train")
    parser.add_argument("--test", default="data/processed/test")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--epochs", type=int, default=15)
    main(parser.parse_args())
