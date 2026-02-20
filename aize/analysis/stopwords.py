"""Stop-word density analysis using NLTK stopwords."""
import re
import nltk

try:
    from nltk.corpus import stopwords as nltk_sw
    _en = set(nltk_sw.words("english"))
    _es = set(nltk_sw.words("spanish"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords as nltk_sw
    _en = set(nltk_sw.words("english"))
    _es = set(nltk_sw.words("spanish"))

STOPWORD_SETS = {"english": _en, "spanish": _es}


def calculate_density(text: str, language: str = "english") -> dict:
    """
    Compute stop-word density for the given text.

    Returns:
        {
          "total_words": int,
          "stop_words":  int,
          "density_pct": float,
          "language": str,
        }
    """
    sw_set = STOPWORD_SETS.get(language, _en)
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return {"total_words": 0, "stop_words": 0, "density_pct": 0.0, "language": language}
    stop_count = sum(1 for w in words if w in sw_set)
    return {
        "total_words": len(words),
        "stop_words": stop_count,
        "density_pct": round(stop_count / len(words) * 100, 4),
        "language": language,
    }
