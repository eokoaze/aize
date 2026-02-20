"""Heap's Law — vocabulary growth (types vs tokens)."""
import re


def analyze_heaps(text: str, sample_every: int = 100) -> dict:
    """
    Compute vocabulary growth as tokens are consumed.

    Returns:
        {
          "tokens": [int, ...],   # token counts sampled every `sample_every`
          "types":  [int, ...],   # unique word counts at each sample
          "total_tokens": int,
          "total_types":  int,
          "diversity_pct": float, # types/tokens * 100
        }
    """
    words = re.findall(r'\b[A-Za-z][a-z]{2,9}\b', text)
    frequency: dict[str, int] = {}
    x, y = [], []
    types = tokens = 0

    for word in words:
        tokens += 1
        word = word.lower()
        if frequency.get(word, 0) == 0:
            types += 1
        frequency[word] = frequency.get(word, 0) + 1
        if tokens % sample_every == 0:
            x.append(tokens)
            y.append(types)

    diversity = round(types / tokens * 100, 2) if tokens > 0 else 0
    return {
        "tokens": x,
        "types": y,
        "total_tokens": tokens,
        "total_types": types,
        "diversity_pct": diversity,
    }
