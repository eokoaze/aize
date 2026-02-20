"""Flesch-Kincaid readability scores."""
import re


def compute_readability(text: str) -> dict:
    """
    Compute Flesch Reading Ease and Flesch-Kincaid Grade Level.

    Returns:
        {
          "flesch_reading_ease": float,    # 0-100, higher = easier
          "fk_grade_level": float,         # US school grade level
          "sentences": int,
          "words": int,
          "syllables": int,
          "interpretation": str,
        }
    """
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    num_sentences = max(len(sentences), 1)
    num_words = max(len(words), 1)

    def count_syllables(word: str) -> int:
        word = word.lower()
        count = len(re.findall(r'[aeiouy]+', word))
        if word.endswith('e') and count > 1:
            count -= 1
        return max(count, 1)

    num_syllables = sum(count_syllables(w) for w in words)

    asl = num_words / num_sentences          # avg sentence length
    asw = num_syllables / num_words          # avg syllables per word

    fre = 206.835 - (1.015 * asl) - (84.6 * asw)
    fkgl = (0.39 * asl) + (11.8 * asw) - 15.59

    if fre >= 90:
        interp = "Very Easy"
    elif fre >= 70:
        interp = "Easy"
    elif fre >= 60:
        interp = "Standard"
    elif fre >= 50:
        interp = "Fairly Difficult"
    elif fre >= 30:
        interp = "Difficult"
    else:
        interp = "Very Confusing"

    return {
        "flesch_reading_ease": round(fre, 2),
        "fk_grade_level":      round(fkgl, 2),
        "sentences":   num_sentences,
        "words":       num_words,
        "syllables":   num_syllables,
        "interpretation": interp,
    }
