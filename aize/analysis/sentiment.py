"""VADER sentiment analysis."""
import nltk

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    _sid = SentimentIntensityAnalyzer()
except LookupError:
    nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    _sid = SentimentIntensityAnalyzer()


def analyze_sentiment(text: str) -> dict:
    """
    Run VADER sentiment analysis on the given text.

    Returns:
        {
          "positive": float,
          "negative": float,
          "neutral":  float,
          "compound": float,   # -1 (most negative) to +1 (most positive)
          "label": str,        # "Positive" | "Negative" | "Neutral"
        }
    """
    scores = _sid.polarity_scores(text)
    if scores["compound"] >= 0.05:
        label = "Positive"
    elif scores["compound"] <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "positive": round(scores["pos"], 4),
        "negative": round(scores["neg"], 4),
        "neutral":  round(scores["neu"], 4),
        "compound": round(scores["compound"], 4),
        "label": label,
    }
