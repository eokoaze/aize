"""
Comprehensive test suite for aize NLP toolkit.
Run with: python test_aize.py
"""
import sys, traceback

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []

SAMPLE_TEXT = """
Natural language processing (NLP) is a subfield of linguistics, computer science,
and artificial intelligence concerned with the interactions between computers and
human language, in particular how to program computers to process and analyze
large amounts of natural language data. The goal is a computer capable of
understanding the contents of documents, including the contextual nuances of
the language within them. The technology can then accurately extract information
and insights contained in the documents, as well as categorize and organize the
documents themselves. Challenges in natural language processing frequently
involve speech recognition, natural language understanding, and natural language
generation. Natural language processing has its roots in the 1950s.
"""

SAMPLE_TEXT_2 = """
Machine learning is a method of data analysis that automates analytical model
building. It is based on the idea that systems can learn from data, identify
patterns and make decisions with minimal human intervention. Machine learning
algorithms are trained on data sets and then applied to new data sets to make
predictions or decisions. Deep learning is a subset of machine learning that
uses neural networks with many layers.
"""

def run_test(name, fn):
    try:
        result = fn()
        results.append((PASS, name, result))
        print(f"{PASS} {name}")
        print(f"       → {result}\n")
    except Exception as e:
        results.append((FAIL, name, str(e)))
        print(f"{FAIL} {name}")
        print(f"       → ERROR: {e}")
        traceback.print_exc()
        print()

# ─────────────────────────────────────────────
# 1. compute_stats
# ─────────────────────────────────────────────
from aize.analysis.stats import compute_stats

def test_stats_keys():
    r = compute_stats(SAMPLE_TEXT)
    assert "lines" in r and "words" in r and "characters" in r and "spaces" in r
    assert r["words"] > 0
    return r

def test_stats_empty():
    r = compute_stats("")
    assert r["words"] == 0
    return r

run_test("compute_stats — basic keys & values", test_stats_keys)
run_test("compute_stats — empty string", test_stats_empty)

# ─────────────────────────────────────────────
# 2. analyze_groupwords
# ─────────────────────────────────────────────
from aize.analysis.groupwords import analyze_groupwords

def test_groupwords_basic():
    r = analyze_groupwords(SAMPLE_TEXT)
    assert isinstance(r, dict)
    assert all(isinstance(k, int) for k in r.keys())
    assert all(isinstance(v, int) for v in r.values())
    assert len(r) > 0
    return r

def test_groupwords_empty():
    r = analyze_groupwords("")
    assert r == {}
    return r

run_test("analyze_groupwords — basic keys & values", test_groupwords_basic)
run_test("analyze_groupwords — empty string", test_groupwords_empty)

# ─────────────────────────────────────────────
# 3. analyze_zipf
# ─────────────────────────────────────────────
from aize.analysis.zipf import analyze_zipf

def test_zipf_keys():
    r = analyze_zipf(SAMPLE_TEXT)
    assert "frequency" in r and "rank_freq" in r
    assert "hapax_pct" in r and "dis_pct" in r and "freq_gt2_pct" in r
    assert 0 <= r["hapax_pct"] <= 100
    assert 0 <= r["dis_pct"] <= 100
    assert 0 <= r["freq_gt2_pct"] <= 100
    assert abs(r["hapax_pct"] + r["dis_pct"] + r["freq_gt2_pct"] - 100.0) < 0.1
    return {k: v for k, v in r.items() if k != "frequency" and k != "rank_freq"}

def test_zipf_empty():
    r = analyze_zipf("")
    assert r["frequency"] == {} and r["rank_freq"] == []
    return r

run_test("analyze_zipf — keys, types, percentages sum to ~100", test_zipf_keys)
run_test("analyze_zipf — empty string", test_zipf_empty)

# ─────────────────────────────────────────────
# 4. analyze_heaps
# ─────────────────────────────────────────────
from aize.analysis.heaps import analyze_heaps

def test_heaps_keys():
    r = analyze_heaps(SAMPLE_TEXT)
    assert "tokens" in r and "types" in r
    assert "total_tokens" in r and "total_types" in r and "diversity_pct" in r
    assert r["total_tokens"] > 0
    assert r["total_types"] <= r["total_tokens"]
    assert 0 <= r["diversity_pct"] <= 100
    return {k: v for k, v in r.items() if k not in ("tokens", "types")}

def test_heaps_empty():
    r = analyze_heaps("")
    assert r["total_tokens"] == 0 and r["total_types"] == 0
    return r

run_test("analyze_heaps — keys, types ≤ tokens, diversity in range", test_heaps_keys)
run_test("analyze_heaps — empty string", test_heaps_empty)

# ─────────────────────────────────────────────
# 5. calculate_density
# ─────────────────────────────────────────────
from aize.analysis.stopwords import calculate_density

def test_density_english():
    r = calculate_density(SAMPLE_TEXT, language="english")
    assert "total_words" in r and "stop_words" in r and "density_pct" in r
    assert r["total_words"] > 0
    assert 0 <= r["density_pct"] <= 100
    assert r["stop_words"] <= r["total_words"]
    return r

def test_density_spanish():
    r = calculate_density("El perro corre y el gato salta.", language="spanish")
    assert r["total_words"] > 0
    return r

def test_density_empty():
    r = calculate_density("")
    assert r["total_words"] == 0 and r["density_pct"] == 0.0
    return r

run_test("calculate_density — English", test_density_english)
run_test("calculate_density — Spanish", test_density_spanish)
run_test("calculate_density — empty string", test_density_empty)

# ─────────────────────────────────────────────
# 6. compare_vocab
# ─────────────────────────────────────────────
from aize.analysis.vocab import compare_vocab
from aize.analysis.zipf import analyze_zipf as _zipf

def test_vocab_compare():
    freq_a = _zipf(SAMPLE_TEXT)["frequency"]
    freq_b = _zipf(SAMPLE_TEXT_2)["frequency"]
    r = compare_vocab(freq_a, "NLP", freq_b, "ML")
    assert "name_a" in r and "name_b" in r
    assert "common" in r and "only_in_a" in r and "only_in_b" in r
    assert r["name_a"] == "NLP" and r["name_b"] == "ML"
    assert r["size_a"] == len(freq_a) and r["size_b"] == len(freq_b)
    assert r["common"] + r["only_in_a"] == r["size_a"]
    return r

def test_vocab_identical():
    freq = _zipf(SAMPLE_TEXT)["frequency"]
    r = compare_vocab(freq, "A", freq, "B")
    assert r["only_in_a"] == 0 and r["only_in_b"] == 0
    assert r["pct_a_missing_from_b"] == 0.0
    return r

run_test("compare_vocab — two different texts", test_vocab_compare)
run_test("compare_vocab — identical texts (zero missing)", test_vocab_identical)

# ─────────────────────────────────────────────
# 7. compute_tfidf & compute_ngrams
# ─────────────────────────────────────────────
from aize.analysis.tfidf import compute_tfidf, compute_ngrams

def test_tfidf_basic():
    r = compute_tfidf([SAMPLE_TEXT, SAMPLE_TEXT_2], ["nlp", "ml"], top_n=5)
    assert "nlp" in r and "ml" in r
    assert len(r["nlp"]) <= 5 and len(r["ml"]) <= 5
    assert all(isinstance(term, str) and isinstance(score, float) for term, score in r["nlp"])
    return {"nlp_top": r["nlp"][:3], "ml_top": r["ml"][:3]}

def test_tfidf_empty():
    r = compute_tfidf([], [], top_n=5)
    assert r == {}
    return r

def test_ngrams_bigrams():
    r = compute_ngrams(SAMPLE_TEXT, n=2, top_n=5)
    assert len(r) <= 5
    assert all(len(ng.split()) == 2 for ng, _ in r)
    return r

def test_ngrams_trigrams():
    r = compute_ngrams(SAMPLE_TEXT, n=3, top_n=5)
    assert all(len(ng.split()) == 3 for ng, _ in r)
    return r

run_test("compute_tfidf — two documents, top 5 terms", test_tfidf_basic)
run_test("compute_tfidf — empty corpus", test_tfidf_empty)
run_test("compute_ngrams — bigrams", test_ngrams_bigrams)
run_test("compute_ngrams — trigrams", test_ngrams_trigrams)

# ─────────────────────────────────────────────
# 8. analyze_sentiment
# ─────────────────────────────────────────────
from aize.analysis.sentiment import analyze_sentiment

def test_sentiment_keys():
    r = analyze_sentiment(SAMPLE_TEXT)
    assert all(k in r for k in ("positive", "negative", "neutral", "compound", "label"))
    assert r["label"] in ("Positive", "Negative", "Neutral")
    assert -1.0 <= r["compound"] <= 1.0
    return r

def test_sentiment_positive():
    r = analyze_sentiment("I absolutely love this! It is amazing, wonderful, and fantastic.")
    assert r["label"] == "Positive"
    return r

def test_sentiment_negative():
    r = analyze_sentiment("This is terrible, horrible, awful and completely dreadful.")
    assert r["label"] == "Negative"
    return r

run_test("analyze_sentiment — keys and range", test_sentiment_keys)
run_test("analyze_sentiment — clearly positive text", test_sentiment_positive)
run_test("analyze_sentiment — clearly negative text", test_sentiment_negative)

# ─────────────────────────────────────────────
# 9. compute_readability
# ─────────────────────────────────────────────
from aize.analysis.readability import compute_readability

def test_readability_keys():
    r = compute_readability(SAMPLE_TEXT)
    assert all(k in r for k in ("flesch_reading_ease", "fk_grade_level", "sentences", "words", "syllables", "interpretation"))
    assert r["words"] > 0 and r["sentences"] > 0 and r["syllables"] > 0
    assert r["interpretation"] in ("Very Easy", "Easy", "Standard", "Fairly Difficult", "Difficult", "Very Confusing")
    return r

def test_readability_simple():
    r = compute_readability("The cat sat on the mat. The dog ran fast.")
    assert r["flesch_reading_ease"] > 60, f"Expected easy text, got {r['flesch_reading_ease']}"
    return r

run_test("compute_readability — keys and valid interpretation", test_readability_keys)
run_test("compute_readability — simple text should be readable", test_readability_simple)

# ─────────────────────────────────────────────
# 10. analyze_pos
# ─────────────────────────────────────────────
from aize.analysis.pos import analyze_pos

def test_pos_keys():
    r = analyze_pos(SAMPLE_TEXT)
    expected_groups = {"Noun", "Verb", "Adjective", "Adverb", "Pronoun", "Determiner", "Preposition", "Conjunction", "Other"}
    assert expected_groups == set(r.keys())
    assert all(isinstance(v, int) and v >= 0 for v in r.values())
    assert r["Noun"] > 0
    return r

def test_pos_empty():
    r = analyze_pos("   ")
    assert all(isinstance(v, int) for v in r.values())
    return r

run_test("analyze_pos — all POS groups present, Noun > 0", test_pos_keys)
run_test("analyze_pos — whitespace-only text", test_pos_empty)

# ─────────────────────────────────────────────
# 11. generate_wordcloud
# ─────────────────────────────────────────────
from aize.analysis.wordcloud_gen import generate_wordcloud

def test_wordcloud_bytes():
    r = generate_wordcloud(SAMPLE_TEXT)
    assert isinstance(r, bytes)
    assert len(r) > 0
    # PNG magic bytes
    assert r[:4] == b'\x89PNG', f"Expected PNG header, got {r[:4]}"
    return f"{len(r)} bytes, valid PNG"

run_test("generate_wordcloud — returns non-empty PNG bytes", test_wordcloud_bytes)

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
print("\n" + "="*60)
passed = sum(1 for s, _, _ in results if s == PASS)
failed = sum(1 for s, _, _ in results if s == FAIL)
print(f"RESULTS: {passed} passed, {failed} failed out of {len(results)} tests")
print("="*60)

if failed > 0:
    print("\nFailed tests:")
    for status, name, detail in results:
        if status == FAIL:
            print(f"  {name}: {detail}")
    sys.exit(1)
