# aize · NLP Analysis Toolkit

> A lightweight, pip-installable Python library for deep text analysis — covering everything from Zipf's law to sentiment, readability, TF-IDF, and more. Comes with a Streamlit dashboard and a FastAPI backend out of the box.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Module Reference](#module-reference)
  - [compute_stats](#compute_stats)
  - [analyze_groupwords](#analyze_groupwords)
  - [analyze_zipf](#analyze_zipf)
  - [analyze_heaps](#analyze_heaps)
  - [calculate_density](#calculate_density)
  - [compare_vocab](#compare_vocab)
  - [compute_tfidf](#compute_tfidf)
  - [compute_ngrams](#compute_ngrams)
  - [analyze_sentiment](#analyze_sentiment)
  - [compute_readability](#compute_readability)
  - [analyze_pos](#analyze_pos)
  - [generate_wordcloud](#generate_wordcloud)
- [Streamlit Dashboard](#streamlit-dashboard)
- [FastAPI Backend](#fastapi-backend)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [License](#license)

---

## Features

| Category | Capability |
|---|---|
| 📊 **Statistics** | Word count, unique words, avg word length, sentence count |
| 📏 **Word Grouping** | Frequency distribution grouped by word length |
| 📉 **Zipf's Law** | Rank-frequency distribution, hapax & dis legomena percentages |
| 📈 **Heap's Law** | Vocabulary growth curve as corpus size increases |
| 🚫 **Stopwords** | Stopword density analysis |
| 🔤 **Vocabulary** | Side-by-side vocabulary comparison across multiple texts |
| 🔍 **TF-IDF** | Top keyword extraction per document in a corpus |
| 🔗 **N-grams** | Most common bigrams and trigrams |
| 💬 **Sentiment** | VADER-based positive / negative / neutral / compound scoring |
| 📖 **Readability** | Flesch Reading Ease & Flesch-Kincaid Grade Level |
| 🏷️ **POS Tagging** | Part-of-speech frequency breakdown |
| ☁️ **Word Cloud** | Generates word cloud images from any text |
| 🖥️ **Dashboard** | Interactive Streamlit UI for all analyses |
| ⚡ **API** | FastAPI REST backend for programmatic access |

---

## Installation

### From source (development)

```bash
git clone https://github.com/<your-username>/aize.git
cd aize
pip install -e .
```

### From PyPI *(coming soon)*

```bash
pip install aize
```

> **Python 3.9+** is required.

---

## Quick Start

```python
import aize

text = """
Natural language processing is a subfield of linguistics and artificial intelligence.
It is primarily concerned with giving computers the ability to understand text and speech.
"""

# Basic stats
print(aize.compute_stats(text))

# Sentiment
print(aize.analyze_sentiment(text))

# Readability
print(aize.compute_readability(text))

# Zipf's Law
print(aize.analyze_zipf(text))
```

---

## Module Reference

### `compute_stats`

```python
from aize import compute_stats

result = compute_stats(text)
```

Returns basic corpus statistics.

| Key | Type | Description |
|---|---|---|
| `word_count` | `int` | Total number of words |
| `unique_words` | `int` | Number of distinct words |
| `avg_word_length` | `float` | Average characters per word |
| `sentence_count` | `int` | Number of sentences |

---

### `analyze_groupwords`

```python
from aize import analyze_groupwords

result = analyze_groupwords(text)
```

Groups words by their character length and returns frequency counts per length bucket.

---

### `analyze_zipf`

```python
from aize import analyze_zipf

result = analyze_zipf(text)
```

Computes Zipf's Law statistics over the text.

| Key | Type | Description |
|---|---|---|
| `frequency` | `dict` | `{word: count}` sorted most → least frequent |
| `rank_freq` | `list[tuple]` | `[(rank, count)]` for rank-frequency plotting |
| `hapax_pct` | `float` | % of words appearing exactly once |
| `dis_pct` | `float` | % of words appearing exactly twice |
| `freq_gt2_pct` | `float` | % of words appearing more than twice |

---

### `analyze_heaps`

```python
from aize import analyze_heaps

result = analyze_heaps(text)
```

Returns a vocabulary growth curve (Heap's Law). Useful for visualising how the vocabulary expands as more text is read.

---

### `calculate_density`

```python
from aize import calculate_density

result = calculate_density(text)
```

Calculates the proportion of stopwords in the text, returning a stopword density percentage and associated word lists.

---

### `compare_vocab`

```python
from aize import compare_vocab

result = compare_vocab({"doc1": text1, "doc2": text2})
```

Compares vocabulary across multiple documents — unique words per document, shared vocabulary, and overlap statistics.

---

### `compute_tfidf`

```python
from aize import compute_tfidf

result = compute_tfidf(
    texts=["text of doc1...", "text of doc2..."],
    labels=["doc1", "doc2"],
    top_n=15
)
# Returns: {"doc1": [("word", score), ...], "doc2": [...]}
```

Extracts the top `n` TF-IDF keywords for each document in a corpus. Uses scikit-learn under the hood with English stopword filtering.

---

### `compute_ngrams`

```python
from aize import compute_ngrams

bigrams  = compute_ngrams(text, n=2, top_n=20)
trigrams = compute_ngrams(text, n=3, top_n=20)
# Returns: [("phrase here", count), ...]
```

Returns the most frequent n-grams (bigrams, trigrams, etc.) from the text.

---

### `analyze_sentiment`

```python
from aize import analyze_sentiment

result = analyze_sentiment(text)
```

Runs VADER sentiment analysis. NLTK's `vader_lexicon` is auto-downloaded on first use.

| Key | Type | Description |
|---|---|---|
| `positive` | `float` | Proportion of positive sentiment |
| `negative` | `float` | Proportion of negative sentiment |
| `neutral` | `float` | Proportion of neutral sentiment |
| `compound` | `float` | Overall score from `-1.0` (most negative) to `+1.0` (most positive) |
| `label` | `str` | `"Positive"`, `"Negative"`, or `"Neutral"` |

---

### `compute_readability`

```python
from aize import compute_readability

result = compute_readability(text)
```

Computes Flesch-Kincaid readability metrics.

| Key | Type | Description |
|---|---|---|
| `flesch_reading_ease` | `float` | 0–100 score; higher = easier to read |
| `fk_grade_level` | `float` | Approximate US school grade level |
| `sentences` | `int` | Sentence count |
| `words` | `int` | Word count |
| `syllables` | `int` | Total syllables |
| `interpretation` | `str` | `"Very Easy"` → `"Very Confusing"` |

---

### `analyze_pos`

```python
from aize import analyze_pos

result = analyze_pos(text)
```

Returns a part-of-speech frequency breakdown (nouns, verbs, adjectives, adverbs, etc.) using NLTK's POS tagger.

---

### `generate_wordcloud`

```python
from aize import generate_wordcloud

image = generate_wordcloud(text)
```

Generates a word cloud image from the input text. Returns a PIL `Image` object that can be displayed or saved.

```python
image.save("wordcloud.png")
```

---

## Streamlit Dashboard

An interactive, browser-based UI for all analyses is included.

```bash
streamlit run nlp_dashboard.py
```

The dashboard lets you upload one or more `.txt` files and interactively explore all analysis modules with charts and tables powered by Plotly.

---

## FastAPI Backend

A REST API is included for programmatic or remote access to the toolkit.

```bash
uvicorn api:app --reload
```

The API will be available at `http://127.0.0.1:8000`. Interactive docs are auto-generated at:

- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc**: `http://127.0.0.1:8000/redoc`

---

## Dependencies

| Package | Purpose |
|---|---|
| `nltk >= 3.8` | Tokenisation, POS tagging, VADER sentiment |
| `scikit-learn >= 1.2` | TF-IDF vectorisation |
| `wordcloud >= 1.9` | Word cloud image generation |
| `pandas >= 1.5` | Data manipulation |
| `plotly >= 5.0` | Interactive charts in the dashboard |
| `streamlit >= 1.28` | Web dashboard UI |
| `fastapi >= 0.100` | REST API framework |
| `uvicorn >= 0.23` | ASGI server for FastAPI |
| `python-multipart >= 0.0.6` | File upload support for FastAPI |

---

## Project Structure

```
aize/
├── aize/                        # Core library package
│   ├── __init__.py              # Public API surface
│   └── analysis/
│       ├── stats.py             # Basic text statistics
│       ├── groupwords.py        # Word length grouping
│       ├── zipf.py              # Zipf's law analysis
│       ├── heaps.py             # Heap's law analysis
│       ├── stopwords.py         # Stopword density
│       ├── vocab.py             # Vocabulary comparison
│       ├── tfidf.py             # TF-IDF & n-grams
│       ├── sentiment.py         # VADER sentiment
│       ├── readability.py       # Flesch-Kincaid scores
│       ├── pos.py               # POS tagging
│       └── wordcloud_gen.py     # Word cloud generation
├── nlp_dashboard.py             # Streamlit dashboard
├── api.py                       # FastAPI REST backend
├── setup.py                     # Package configuration
├── requirements.txt             # Dependencies
└── README.md
```

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

<p align="center">Built with ❤️ using Python, NLTK, scikit-learn, Streamlit & FastAPI</p>
