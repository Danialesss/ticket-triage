"""Load saved artifacts and classify new ticket text.

CLI:
    python -m src.predict "I was charged twice this month"
"""
import argparse
import json
import os

import torch

from src.dataset import encode
from src.model import TextClassifier


def load_artifacts(path: str = "artifacts"):
    with open(os.path.join(path, "vocab.json")) as f:
        vocab = json.load(f)
    with open(os.path.join(path, "labels.json")) as f:
        id2label = {int(k): v for k, v in json.load(f).items()}
    model = TextClassifier(len(vocab), len(id2label))
    model.load_state_dict(torch.load(os.path.join(path, "model.pt"), map_location="cpu"))
    model.eval()
    return model, vocab, id2label


def predict(text: str, model, vocab, id2label):
    ids = torch.tensor([encode(text, vocab)], dtype=torch.long)
    with torch.no_grad():
        probs = torch.softmax(model(ids), dim=1)[0]
    idx = int(probs.argmax())
    return id2label[idx], float(probs[idx])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()
    model, vocab, id2label = load_artifacts(args.artifacts)
    label, confidence = predict(args.text, model, vocab, id2label)
    print(f"{label}  ({confidence:.1%})")
