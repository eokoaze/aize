"""TF-IDF keyword extraction and N-gram frequency analysis."""
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer


def compute_tfidf(texts: list[str], labels: list[str], top_n: int = 15) -> dict:
    """
    Compute TF-IDF scores across a corpus of texts.

    Args:
        texts:  list of raw text strings
        labels: matching list of file/document names
        top_n:  number of top keywords to return per document

    Returns:
        {label: [(term, score), ...], ...}
    """
    if not texts:
        return {}
    vec = TfidfVectorizer(stop_words="english", max_features=5000,
                          token_pattern=r'\b[a-z]{3,}\b')
    try:
        matrix = vec.fit_transform(texts)
    except ValueError:
        return {}

    terms = vec.get_feature_names_out()
    result = {}
    for idx, label in enumerate(labels):
        scores = matrix[idx].toarray().flatten()
        top_idx = scores.argsort()[::-1][:top_n]
        result[label] = [(terms[i], round(float(scores[i]), 4)) for i in top_idx if scores[i] > 0]
    return result


def compute_ngrams(text: str, n: int = 2, top_n: int = 20) -> list[tuple]:
    """
    Return the most common n-grams (bigrams or trigrams) in text.

    Returns:
        [(ngram_string, count), ...]
    """
    words = re.findall(r'\b[a-z]{2,}\b', text.lower())
    ngrams = zip(*[words[i:] for i in range(n)])
    counts = Counter(" ".join(g) for g in ngrams)
    return counts.most_common(top_n)
