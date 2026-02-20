"""Zipf's Law analysis — rank/frequency + hapax/dis legomena."""
import re


def analyze_zipf(text: str) -> dict:
    """
    Compute word frequency distribution and Zipf statistics.

    Returns:
        {
          "frequency": {word: count, ...},    # sorted most→least frequent
          "rank_freq": [(rank, count), ...],   # for rank-frequency plot
          "hapax_pct": float,                  # % words appearing once
          "dis_pct":   float,                  # % words appearing twice
          "freq_gt2_pct": float,               # % words appearing > 2 times
        }
    """
    words = re.findall(r'\b[A-Za-z][a-z]{2,9}\b', text)
    frequency: dict[str, int] = {}
    for word in words:
        frequency[word] = frequency.get(word, 0) + 1

    total = len(frequency)
    if total == 0:
        return {"frequency": {}, "rank_freq": [], "hapax_pct": 0, "dis_pct": 0, "freq_gt2_pct": 0}

    hapax = sum(1 for c in frequency.values() if c == 1)
    dis   = sum(1 for c in frequency.values() if c == 2)
    gt2   = total - hapax - dis

    sorted_freq = dict(sorted(frequency.items(), key=lambda x: x[1], reverse=True))
    rank_freq = list(enumerate(sorted_freq.values(), start=1))

    return {
        "frequency":   sorted_freq,
        "rank_freq":   rank_freq,
        "hapax_pct":   round(hapax / total * 100, 2),
        "dis_pct":     round(dis   / total * 100, 2),
        "freq_gt2_pct": round(gt2  / total * 100, 2),
    }
