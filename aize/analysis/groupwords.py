"""Word-length (groupwords) distribution analysis."""
import re


def analyze_groupwords(text: str) -> dict:
    """
    Group unique words by their character length.
    Returns a dict: {length: count_of_unique_words}
    """
    groups: dict[int, set] = {}
    words = text.split()
    for word in words:
        word = word.lower().strip('.,!?"():;[]#«»')
        if not word:
            continue
        size = len(word)
        if size not in groups:
            groups[size] = set()
        groups[size].add(word)

    return {size: len(words_set) for size, words_set in sorted(groups.items())}
