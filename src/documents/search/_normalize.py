from __future__ import annotations

import unicodedata


def ascii_fold(text: str) -> str:
    """Normalize unicode text to ASCII equivalents for search consistency."""
    return unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode()


def word_trigrams_text(text: str) -> str:
    """Pre-process text into space-separated per-word trigrams for whitespace-tokenizer indexing.

    Splits on whitespace, lowercases and ascii-folds each word, then generates
    character trigrams. Words shorter than 3 chars produce no trigrams and are
    dropped — consistent with the query side which ignores sub-3-char tokens.
    """
    tokens = []
    for word in text.split():
        w = ascii_fold(word.lower())
        if len(w) >= 3:
            tokens.extend(w[i : i + 3] for i in range(len(w) - 2))
    return " ".join(tokens)
