"""
aize NLP Dashboard
Run with: streamlit run nlp_dashboard.py
"""
import sys, os, io
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from langdetect import detect as _langdetect
except ImportError:
    _langdetect = None
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from PIL import Image

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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="aize · NLP Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0f1117; }
  [data-testid="stSidebar"] { background: #161b27; border-right: 1px solid #2a2f3e; }
  .metric-card {
    background: #1e2233; border-radius: 12px; padding: 18px 22px;
    border: 1px solid #2e3450; margin-bottom: 10px;
  }
  .metric-value { font-size: 2rem; font-weight: 700; color: #7c8dfc; line-height: 1; }
  .metric-label { font-size: 0.82rem; color: #8892a4; margin-top: 4px; }
  h1, h2, h3 { color: #e8ecf4 !important; }
  .stTabs [data-baseweb="tab"] { color: #8892a4; font-size: 0.9rem; }
  .stTabs [aria-selected="true"] { color: #7c8dfc !important; border-bottom-color: #7c8dfc !important; }
  .pill {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600; margin: 2px;
  }
  .pill-pos  { background:#2a3a5e; color:#7c8dfc; }
  .pill-neg  { background:#3a2a2a; color:#f47c7c; }
  .pill-neu  { background:#2a3a2a; color:#7cf498; }
</style>
""", unsafe_allow_html=True)

# PX_THEME: safe to spread into px.*() function calls (template only).
# PLOTLY_LAYOUT: full layout properties for fig.update_layout() calls.
PX_THEME = dict(template="plotly_dark")
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#1e2233",
    plot_bgcolor="#1e2233",
    font_color="#c8cfe0",
)


def _theme(fig):
    """Apply the dark layout theme to any plotly figure and return it."""
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


# Supported languages and their langdetect ISO codes
_SUPPORTED = ["english", "spanish"]
_LANG_CODE_MAP = {"en": "english", "es": "spanish"}


def _detect_language(text: str) -> str:
    """Auto-detect language from a text sample. Falls back to 'english'."""
    if _langdetect is None or not text.strip():
        return "english"
    try:
        code = _langdetect(text[:3_000])
        return _LANG_CODE_MAP.get(code, "english")
    except Exception:
        return "english"


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 aize NLP")
    st.markdown("*Upload text files to analyse*")
    st.markdown("---")
    uploaded = st.file_uploader(
        "Upload .txt files", type=["txt"], accept_multiple_files=True
    )
    lang_map = {}  # keyed by (index, name) to survive duplicate filenames
    if uploaded:
        st.markdown("**Language per file** *(auto-detected — override if needed)*")
        for i, f in enumerate(uploaded):
            # Peek at first 3 000 chars for detection, then reset pointer
            sample = f.read(3_000).decode("utf-8", errors="ignore")
            f.seek(0)
            detected = _detect_language(sample)
            lang_map[(i, f.name)] = st.selectbox(
                f.name,
                _SUPPORTED,
                index=_SUPPORTED.index(detected),
                key=f"lang_{i}_{f.name}",
                help=f"Auto-detected: {detected}",
            )
    st.markdown("---")
    st.markdown("**aize v0.1.0**  \n[API docs →](http://localhost:8000/docs)")

# ── No files ──────────────────────────────────────────────────────────────────
if not uploaded:
    st.markdown("# 🔬 aize NLP Dashboard")
    st.info("👈 Upload one or more `.txt` files in the sidebar to get started.")
    st.markdown("""
    **What you'll see:**
    - 📊 Overview stats · Word distribution · Zipf's Law · Heap's Law
    - 🛑 Stop-words · 🔤 Vocab comparison · 🏷️ TF-IDF · Ngrams
    - 💬 Sentiment · 🧩 POS tags · 📖 Readability · ☁️ Word cloud
    """)
    st.stop()

# ── Load all files ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Analysing…")
def load_file(name: str, content: bytes, language: str):
    text = content.decode("utf-8", errors="ignore")

    # Run all independent analyses in parallel for maximum speed.
    tasks = {
        "stats":       lambda: compute_stats(text),
        "groupwords":  lambda: analyze_groupwords(text),
        "zipf":        lambda: analyze_zipf(text),
        "heaps":       lambda: analyze_heaps(text),
        "stopwords":   lambda: calculate_density(text, language),
        "sentiment":   lambda: analyze_sentiment(text),
        "readability": lambda: compute_readability(text),
        "pos":         lambda: analyze_pos(text),
    }
    results = {}
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = {pool.submit(fn): key for key, fn in tasks.items()}
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    return {"name": name, "text": text, "language": language, **results}

files_data = {}
_name_counts: dict[str, int] = {}
load_errors = []

for i, f in enumerate(uploaded):
    # Deduplicate display names: "file.txt", "file.txt (2)", "file.txt (3)", …
    _name_counts[f.name] = _name_counts.get(f.name, 0) + 1
    display_name = f.name if _name_counts[f.name] == 1 else f"{f.name} ({_name_counts[f.name]})"

    language = lang_map.get((i, f.name), "english")
    content = f.read()
    try:
        data = load_file(display_name, content, language)
        files_data[display_name] = data
    except Exception as exc:
        load_errors.append((display_name, str(exc)))

# Show friendly error messages instead of raw tracebacks
if load_errors:
    for fname, msg in load_errors:
        st.error(f"⚠️ Could not analyse **{fname}**: {msg}")

if not files_data:
    st.warning("No files could be analysed. Please check your uploads and try again.")
    st.stop()

names = list(files_data.keys())
texts = [d["text"] for d in files_data.values()]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Overview", "📏 Word Dist.", "📈 Zipf's Law", "📉 Heap's Law",
    "🛑 Stop Words", "🔤 Vocab Compare", "🏷️ TF-IDF",
    "🔗 N-grams", "💬 Sentiment", "🧩 POS Tags", "📖 Readability", "☁️ Word Cloud"
])

# ── 1. OVERVIEW ───────────────────────────────────────────────────────────────
with tabs[0]:
    st.header("Overview")
    rows = []
    for name, d in files_data.items():
        s = d["stats"]
        rows.append({
            "File": name,
            "Words": f"{s['words']:,}",
            "Lines": f"{s['lines']:,}",
            "Characters": f"{s['characters']:,}",
            "Spaces": f"{s['spaces']:,}",
            "Language": d["language"].capitalize(),
        })
    st.dataframe(pd.DataFrame(rows), width='stretch')

    # Metric cards
    cols = st.columns(len(files_data))
    for col, (name, d) in zip(cols, files_data.items()):
        with col:
            s = d["stats"]
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{name}</div>
              <div class="metric-value">{s['words']:,}</div>
              <div class="metric-label">words</div>
            </div>""", unsafe_allow_html=True)

# ── 2. WORD DISTRIBUTION ──────────────────────────────────────────────────────
with tabs[1]:
    st.header("Word-Length Distribution")
    sel = st.selectbox("Select file", names, key="wd_sel")
    gw = files_data[sel]["groupwords"]
    fig = px.bar(
        x=list(gw.keys()), y=list(gw.values()),
        labels={"x": "Word Length (chars)", "y": "Unique Words"},
        title=f"Word-Length Distribution — {sel}",
        color=list(gw.values()), color_continuous_scale="Blues",
        **PX_THEME
    )
    fig.update_layout(coloraxis_showscale=False, **PLOTLY_LAYOUT)
    st.plotly_chart(fig, width='stretch')

# ── 3. ZIPF'S LAW ─────────────────────────────────────────────────────────────
with tabs[2]:
    st.header("Zipf's Law")
    fig = go.Figure()
    for name, d in files_data.items():
        rf = d["zipf"]["rank_freq"][:500]
        fig.add_trace(go.Scatter(
            x=[r for r, _ in rf], y=[f for _, f in rf],
            mode="lines", name=name
        ))
    fig.update_layout(
        xaxis_type="log", yaxis_type="log",
        xaxis_title="Rank", yaxis_title="Frequency",
        title="Zipf's Law (log-log)", **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig, width='stretch')

    st.subheader("Hapax / Dis Legomena")
    rows = []
    for name, d in files_data.items():
        z = d["zipf"]
        rows.append({
            "File": name,
            "Hapax legomena (once) %": z["hapax_pct"],
            "Dis legomena (twice) %": z["dis_pct"],
            "Freq > 2 %": z["freq_gt2_pct"],
        })
    st.dataframe(pd.DataFrame(rows), width='stretch')

# ── 4. HEAP'S LAW ─────────────────────────────────────────────────────────────
with tabs[3]:
    st.header("Heap's Law — Vocabulary Growth")
    fig = go.Figure()
    for name, d in files_data.items():
        h = d["heaps"]
        fig.add_trace(go.Scatter(
            x=h["tokens"], y=h["types"], mode="lines", name=name
        ))
    fig.update_layout(
        xaxis_type="log", yaxis_type="log",
        xaxis_title="Tokens", yaxis_title="Unique Types",
        title="Heap's Law (log-log)", **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig, width='stretch')

    rows = [{"File": n, "Tokens": d["heaps"]["total_tokens"],
             "Types": d["heaps"]["total_types"],
             "Diversity %": d["heaps"]["diversity_pct"]}
            for n, d in files_data.items()]
    st.dataframe(pd.DataFrame(rows), width='stretch')

# ── 5. STOP WORDS ─────────────────────────────────────────────────────────────
with tabs[4]:
    st.header("Stop-Word Density")
    rows = [{"File": n, "Total Words": d["stopwords"]["total_words"],
             "Stop Words": d["stopwords"]["stop_words"],
             "Density %": d["stopwords"]["density_pct"],
             "Language": d["language"].capitalize()}
            for n, d in files_data.items()]
    st.dataframe(pd.DataFrame(rows), width='stretch')

    df_sw = pd.DataFrame(rows)
    fig = px.bar(df_sw, x="File", y="Density %", color="Density %",
                 color_continuous_scale="Reds", title="Stop-Word Density (%)",
                 **PX_THEME)
    fig.update_layout(coloraxis_showscale=False, **PLOTLY_LAYOUT)
    st.plotly_chart(fig, width='stretch')

# ── 6. VOCAB COMPARISON ───────────────────────────────────────────────────────
with tabs[5]:
    st.header("Vocabulary Comparison")
    if len(names) < 2:
        st.info("Upload at least 2 files to compare vocabularies.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            a = st.selectbox("File A", names, key="vc_a")
        with col2:
            b_opts = [n for n in names if n != a]
            b = st.selectbox("File B", b_opts, key="vc_b")

        result = compare_vocab(
            files_data[a]["zipf"]["frequency"], a,
            files_data[b]["zipf"]["frequency"], b,
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Unique in A", f"{result['size_a']:,}")
        c2.metric("Unique in B", f"{result['size_b']:,}")
        c3.metric("Common", f"{result['common']:,}")
        c4.metric("% of A missing from B", f"{result['pct_a_missing_from_b']}%")

        fig = px.bar(
            x=["Only in A", "Common", "Only in B"],
            y=[result["only_in_a"], result["common"], result["only_in_b"]],
            labels={"x": "", "y": "Word Count"},
            color=["Only in A", "Common", "Only in B"],
            title=f"Vocabulary Overlap: {a} vs {b}",
            **PX_THEME
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, width='stretch')

# ── 7. TF-IDF ─────────────────────────────────────────────────────────────────
with tabs[6]:
    st.header("TF-IDF — Top Keywords")
    top_n = st.slider("Top N keywords", 5, 30, 15, key="tfidf_n")
    tfidf_results = compute_tfidf(texts, names, top_n=top_n)

    for name, kws in tfidf_results.items():
        if not kws:
            continue
        terms, scores = zip(*kws) if kws else ([], [])
        fig = px.bar(x=list(scores), y=list(terms), orientation="h",
                     title=f"TF-IDF Keywords — {name}",
                     labels={"x": "TF-IDF Score", "y": "Term"},
                     color=list(scores), color_continuous_scale="Blues",
                     **PX_THEME)
        fig.update_layout(yaxis={"autorange": "reversed"}, coloraxis_showscale=False,
                          **PLOTLY_LAYOUT)
        st.plotly_chart(fig, width='stretch')

# ── 8. N-GRAMS ────────────────────────────────────────────────────────────────
with tabs[7]:
    st.header("N-gram Analysis")
    sel_ng = st.selectbox("Select file", names, key="ng_sel")
    col1, col2 = st.columns(2)

    with col1:
        bigrams = compute_ngrams(files_data[sel_ng]["text"], n=2, top_n=15)
        if bigrams:
            bg_df = pd.DataFrame(bigrams, columns=["Bigram", "Count"])
            fig = px.bar(bg_df, x="Count", y="Bigram", orientation="h",
                         title="Top Bigrams", **PX_THEME)
            fig.update_layout(yaxis={"autorange": "reversed"}, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, width='stretch')

    with col2:
        trigrams = compute_ngrams(files_data[sel_ng]["text"], n=3, top_n=15)
        if trigrams:
            tg_df = pd.DataFrame(trigrams, columns=["Trigram", "Count"])
            fig = px.bar(tg_df, x="Count", y="Trigram", orientation="h",
                         title="Top Trigrams", **PX_THEME)
            fig.update_layout(yaxis={"autorange": "reversed"}, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, width='stretch')

# ── 9. SENTIMENT ──────────────────────────────────────────────────────────────
with tabs[8]:
    st.header("Sentiment Analysis (VADER)")
    rows = []
    for name, d in files_data.items():
        s = d["sentiment"]
        rows.append({"File": name, "Label": s["label"],
                     "Positive": s["positive"], "Negative": s["negative"],
                     "Neutral": s["neutral"], "Compound": s["compound"]})
    st.dataframe(pd.DataFrame(rows), width='stretch')

    df_sent = pd.DataFrame(rows)
    fig = go.Figure()
    for col, color in [("Positive", "#7cf498"), ("Negative", "#f47c7c"), ("Neutral", "#7c8dfc")]:
        fig.add_trace(go.Bar(name=col, x=df_sent["File"], y=df_sent[col], marker_color=color))
    fig.update_layout(barmode="stack", title="Sentiment Breakdown per File", **PLOTLY_LAYOUT)
    st.plotly_chart(fig, width='stretch')

    # Compound gauge per file
    for name, d in files_data.items():
        compound = d["sentiment"]["compound"]
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=compound,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": name, "font": {"color": "#c8cfe0"}},
            gauge={
                "axis": {"range": [-1, 1]},
                "bar": {"color": "#7c8dfc"},
                "steps": [
                    {"range": [-1, -0.05], "color": "#3a2a2a"},
                    {"range": [-0.05, 0.05], "color": "#2a2a3a"},
                    {"range": [0.05, 1], "color": "#2a3a2a"},
                ],
                "threshold": {"line": {"color": "white", "width": 2}, "value": compound},
            },
        ))
        fig.update_layout(height=250, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, width='stretch')

# ── 10. POS TAGS ──────────────────────────────────────────────────────────────
with tabs[9]:
    st.header("Part-of-Speech Distribution")
    rows = []
    for name, d in files_data.items():
        row = {"File": name}
        row.update(d["pos"])
        rows.append(row)
    df_pos = pd.DataFrame(rows).set_index("File")
    pos_cols = [c for c in df_pos.columns if c != "File"]
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, tag in enumerate(pos_cols):
        fig.add_trace(go.Bar(name=tag, x=df_pos.index.tolist(),
                             y=df_pos[tag].tolist(),
                             marker_color=colors[i % len(colors)]))
    fig.update_layout(barmode="stack", title="POS Tag Distribution", **PLOTLY_LAYOUT)
    st.plotly_chart(fig, width='stretch')
    st.dataframe(df_pos, width='stretch')

# ── 11. READABILITY ───────────────────────────────────────────────────────────
with tabs[10]:
    st.header("Readability Scores")
    rows = []
    for name, d in files_data.items():
        r = d["readability"]
        rows.append({
            "File": name,
            "Flesch Reading Ease": r["flesch_reading_ease"],
            "FK Grade Level": r["fk_grade_level"],
            "Interpretation": r["interpretation"],
            "Sentences": r["sentences"],
            "Words": r["words"],
            "Syllables": r["syllables"],
        })
    st.dataframe(pd.DataFrame(rows), width='stretch')

    df_r = pd.DataFrame(rows)
    fig = px.bar(df_r, x="File", y="Flesch Reading Ease",
                 color="Flesch Reading Ease", color_continuous_scale="Greens",
                 title="Flesch Reading Ease (higher = easier)", **PX_THEME)
    fig.update_layout(coloraxis_showscale=False, **PLOTLY_LAYOUT)
    st.plotly_chart(fig, width='stretch')

    fig2 = px.bar(df_r, x="File", y="FK Grade Level",
                  color="FK Grade Level", color_continuous_scale="Oranges",
                  title="Flesch-Kincaid Grade Level", **PX_THEME)
    fig2.update_layout(coloraxis_showscale=False, **PLOTLY_LAYOUT)
    st.plotly_chart(fig2, width='stretch')

# ── 12. WORD CLOUD ────────────────────────────────────────────────────────────
with tabs[11]:
    st.header("Word Cloud")
    sel_wc = st.selectbox("Select file", names, key="wc_sel")
    with st.spinner("Generating word cloud…"):
        png = generate_wordcloud(files_data[sel_wc]["text"])
    img = Image.open(io.BytesIO(png))
    st.image(img, caption=sel_wc, width='stretch')
