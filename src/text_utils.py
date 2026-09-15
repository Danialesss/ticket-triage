"""Pure-Python text helpers shared by the Spark prep and the PyTorch dataset.

Kept dependency-free on purpose so the unit tests can run in CI without
installing Spark or Torch.
"""
import re

_non_alnum = re.compile(r"[^a-z0-9\s]")
_whitespace = re.compile(r"\s+")


def clean_text(text) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""
    if text is None:
        return ""
    text = str(text).lower()
    text = _non_alnum.sub(" ", text)
    text = _whitespace.sub(" ", text).strip()
    return text


def simple_tokenize(text):
    """Clean then split on whitespace into a list of tokens."""
    cleaned = clean_text(text)
    return cleaned.split() if cleaned else []
