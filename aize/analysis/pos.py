"""Part-of-speech tag distribution using NLTK."""
import re
import nltk

# Support both old (averaged_perceptron_tagger) and new (averaged_perceptron_tagger_eng)
# NLTK versions — download whichever is missing.
for _resource in ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"):
    try:
        nltk.data.find(f"taggers/{_resource}")
    except LookupError:
        nltk.download(_resource, quiet=True)

for _resource in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{_resource}")
    except LookupError:
        nltk.download(_resource, quiet=True)

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
