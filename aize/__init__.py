"""
aize — NLP Analysis Toolkit
A lightweight, pip-installable library for text analysis.
"""

from aize.analysis.stats import compute_stats
from aize.analysis.groupwords import analyze_groupwords
from aize.analysis.zipf import analyze_zipf
from aize.analysis.heaps import analyze_heaps
from aize.analysis.stopwords import calculate_density
from aize.analysis.vocab import compare_vocab
from aize.analysis.tfidf import compute_tfidf, compute_ngrams
from aize.analysis.sentiment import analyze_sentiment
from aize.analysis.readability import compute_readability
from aize.analysis.pos import analyze_pos
from aize.analysis.wordcloud_gen import generate_wordcloud

__version__ = "0.1.0"
__all__ = [
    "compute_stats",
    "analyze_groupwords",
    "analyze_zipf",
    "analyze_heaps",
    "calculate_density",
    "compare_vocab",
    "compute_tfidf",
    "compute_ngrams",
    "analyze_sentiment",
    "compute_readability",
    "analyze_pos",
    "generate_wordcloud",
]
