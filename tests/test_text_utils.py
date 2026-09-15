from src.text_utils import clean_text, simple_tokenize


def test_clean_text_lowercases_and_strips_punctuation():
    assert clean_text("Hello, WORLD!!!") == "hello world"


def test_clean_text_handles_none():
    assert clean_text(None) == ""


def test_clean_text_collapses_whitespace():
    assert clean_text("too    many\tspaces\n") == "too many spaces"


def test_simple_tokenize():
    assert simple_tokenize("Can't log in!") == ["can", "t", "log", "in"]


def test_simple_tokenize_empty():
    assert simple_tokenize("   ") == []
