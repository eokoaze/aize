"""Vocabulary comparison across multiple texts (zero-frequency problem)."""


def compare_vocab(dict_a: dict, name_a: str, dict_b: dict, name_b: str) -> dict:
    """
    Compare vocabulary between two frequency dictionaries.

    Returns:
        {
          "name_a": str, "size_a": int,
          "name_b": str, "size_b": int,
          "common": int,
          "only_in_a": int,
          "only_in_b": int,
          "pct_a_missing_from_b": float,
        }
    """
    set_a = set(dict_a.keys())
    set_b = set(dict_b.keys())
    common = set_a & set_b
    only_a = set_a - set_b
    only_b = set_b - set_a
    pct_missing = round(len(only_a) / len(set_a) * 100, 2) if set_a else 0.0

    return {
        "name_a": name_a,
        "size_a": len(set_a),
        "name_b": name_b,
        "size_b": len(set_b),
        "common": len(common),
        "only_in_a": len(only_a),
        "only_in_b": len(only_b),
        "pct_a_missing_from_b": pct_missing,
    }
