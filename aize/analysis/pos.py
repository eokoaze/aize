"""Part-of-speech tag distribution using NLTK."""
import re
import nltk

try:
    nltk.data.find("taggers/averaged_perceptron_tagger")
except LookupError:
    nltk.download("averaged_perceptron_tagger", quiet=True)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

# POS tag groups (Penn Treebank)
_TAG_GROUPS = {
    "Noun":         {"NN", "NNS", "NNP", "NNPS"},
    "Verb":         {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"},
    "Adjective":    {"JJ", "JJR", "JJS"},
    "Adverb":       {"RB", "RBR", "RBS"},
    "Pronoun":      {"PRP", "PRP$", "WP", "WP$"},
    "Determiner":   {"DT"},
    "Preposition":  {"IN"},
    "Conjunction":  {"CC"},
    "Other":        set(),
}


def analyze_pos(text: str) -> dict:
    """
    Return grouped POS tag counts.

    Returns:
        {"Noun": int, "Verb": int, "Adjective": int, ...}
    """
    # Work on a sample to keep it fast for large files
    sample = text[:50_000]
    try:
        tokens = nltk.word_tokenize(sample)
    except Exception:
        tokens = sample.split()
    tagged = nltk.pos_tag(tokens)

    counts = {group: 0 for group in _TAG_GROUPS}
    for _, tag in tagged:
        matched = False
        for group, tag_set in _TAG_GROUPS.items():
            if tag in tag_set:
                counts[group] += 1
                matched = True
                break
        if not matched:
            counts["Other"] += 1

    return counts
