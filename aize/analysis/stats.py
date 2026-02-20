"""Word/line/char/space statistics for a text."""
import re


def compute_stats(text: str) -> dict:
    """Return basic text statistics."""
    lines = text.splitlines()
    words = text.split()
    chars = sum(1 for c in text if c not in (" ", "\n"))
    spaces = text.count(" ")
    return {
        "lines": len(lines),
        "words": len(words),
        "characters": chars,
        "spaces": spaces,
    }
