"""
aize REST API
Run with: uvicorn api:app --reload
Interactive docs at: http://localhost:8000/docs
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
import sys, os

# Make sure aize is importable from this directory
sys.path.insert(0, os.path.dirname(__file__))

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

app = FastAPI(
    title="aize NLP API",
    description="REST API for the aize NLP analysis toolkit. Upload text files or send raw text to receive structured NLP analysis results.",
    version="0.1.0",
)


# ── Helpers ──────────────────────────────────────────────────────────────────

async def _read_file(file: UploadFile) -> str:
    raw = await file.read()
    return raw.decode("utf-8", errors="ignore")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "aize NLP API", "version": "0.1.0"}


@app.post("/analyze/sentiment", tags=["Analysis"],
          summary="VADER sentiment analysis on raw text")
def sentiment(text: str = Form(..., description="Raw text to analyse")):
    """Returns VADER positive / negative / neutral / compound scores and a label."""
    return analyze_sentiment(text)


@app.post("/analyze/readability", tags=["Analysis"],
          summary="Flesch-Kincaid readability scores")
def readability(text: str = Form(...)):
    """Returns Flesch Reading Ease, FK Grade Level, and an interpretation label."""
    return compute_readability(text)


@app.post("/analyze/ngrams", tags=["Analysis"],
          summary="Top N-grams (bigrams or trigrams) in a text")
def ngrams(
    text: str = Form(...),
    n: int = Form(2, description="N-gram size: 2 = bigrams, 3 = trigrams"),
    top_n: int = Form(20, description="Number of top n-grams to return"),
):
    results = compute_ngrams(text, n=n, top_n=top_n)
    return {"n": n, "top_ngrams": [{"ngram": ng, "count": c} for ng, c in results]}


@app.post("/analyze/zipf", tags=["Analysis"],
          summary="Zipf's Law + hapax/dis legomena statistics")
async def zipf(file: UploadFile = File(...)):
    """Upload a .txt file. Returns rank-frequency data and hapax/dis-legomena percentages."""
    text = await _read_file(file)
    result = analyze_zipf(text)
    # Trim rank_freq for JSON response (keep first 1000 points)
    result["rank_freq"] = result["rank_freq"][:1000]
    return result


@app.post("/analyze/heaps", tags=["Analysis"],
          summary="Heap's Law — vocabulary growth over tokens")
async def heaps(file: UploadFile = File(...)):
    """Upload a .txt file. Returns types/tokens growth series and diversity %."""
    text = await _read_file(file)
    return analyze_heaps(text)


@app.post("/analyze/stopwords", tags=["Analysis"],
          summary="Stop-word density analysis")
async def stopwords(
    file: UploadFile = File(...),
    language: str = Form("english", description="'english' or 'spanish'"),
):
    """Upload a .txt file. Returns total words, stop-word count, and density %."""
    text = await _read_file(file)
    return calculate_density(text, language=language)


@app.post("/analyze/tfidf", tags=["Analysis"],
          summary="TF-IDF top keywords for uploaded files")
async def tfidf(
    files: list[UploadFile] = File(...),
    top_n: int = Form(15, description="Number of top keywords per document"),
):
    """Upload one or more .txt files. Returns top TF-IDF keywords per file."""
    texts, labels = [], []
    for f in files:
        texts.append(await _read_file(f))
        labels.append(f.filename)
    result = compute_tfidf(texts, labels, top_n=top_n)
    return {label: [{"term": t, "score": s} for t, s in kws] for label, kws in result.items()}


@app.post("/analyze/wordcloud", tags=["Analysis"],
          summary="Generate a word cloud PNG image")
async def wordcloud(file: UploadFile = File(...)):
    """Upload a .txt file. Returns a PNG image of the word cloud."""
    text = await _read_file(file)
    png_bytes = generate_wordcloud(text)
    return Response(content=png_bytes, media_type="image/png")


@app.post("/analyze/full", tags=["Analysis"],
          summary="Run ALL analyses on an uploaded file")
async def full_analysis(
    file: UploadFile = File(...),
    language: str = Form("english", description="'english' or 'spanish'"),
):
    """
    Upload a single .txt file and receive every analysis result in one response.
    (Word cloud is excluded from this endpoint — use /analyze/wordcloud for images.)
    """
    text = await _read_file(file)
    zipf_result = analyze_zipf(text)
    zipf_result["rank_freq"] = zipf_result["rank_freq"][:500]

    return {
        "filename":      file.filename,
        "language":      language,
        "stats":         compute_stats(text),
        "stopwords":     calculate_density(text, language=language),
        "sentiment":     analyze_sentiment(text),
        "readability":   compute_readability(text),
        "zipf":          zipf_result,
        "heaps":         analyze_heaps(text),
        "groupwords":    analyze_groupwords(text),
        "pos_tags":      analyze_pos(text),
        "tfidf":         compute_tfidf([text], [file.filename or "file"], top_n=15),
        "bigrams":       [{"ngram": g, "count": c} for g, c in compute_ngrams(text, n=2, top_n=15)],
        "trigrams":      [{"ngram": g, "count": c} for g, c in compute_ngrams(text, n=3, top_n=15)],
    }
