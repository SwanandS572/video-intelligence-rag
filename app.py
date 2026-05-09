import sys
sys.setrecursionlimit(10000)  # Fix: maximum recursion depth exceeded on joblib.load

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from google import genai
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="RAG Video Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body { font-family: 'Inter', sans-serif; }
.stApp { font-family: 'Inter', sans-serif; }

.block-container {
    padding: 2rem 3rem 4rem !important;
    max-width: 1140px !important;
}
section[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1.25rem !important;
}
.rag-header {
    padding-bottom: 1.25rem;
    margin-bottom: 1.75rem;
    border-bottom: 1px solid rgba(128,128,128,0.15);
}
.rag-title {
    font-size: 1.75rem; font-weight: 700;
    letter-spacing: -0.03em; margin: 0 0 0.3rem; color: inherit;
}
.rag-subtitle {
    font-size: 0.92rem; opacity: 0.5; margin: 0; font-weight: 400;
}
.status-pill {
    display: inline-flex; align-items: center; gap: 7px;
    background: rgba(34,197,94,0.1); color: #22c55e;
    border: 1px solid rgba(34,197,94,0.25); border-radius: 999px;
    padding: 5px 14px; font-size: 0.78rem; font-weight: 600;
    margin-bottom: 1.5rem; letter-spacing: 0.01em;
}
.status-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #22c55e; box-shadow: 0 0 6px #22c55e;
    display: inline-block; flex-shrink: 0;
}
.stats-grid {
    display: grid; grid-template-columns: repeat(3,1fr);
    gap: 10px; margin-bottom: 2rem;
}
.stat-box {
    border: 1px solid rgba(128,128,128,0.18);
    border-radius: 12px; padding: 1.1rem 1.25rem;
    background: rgba(128,128,128,0.04);
}
.stat-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.7rem; font-weight: 700;
    letter-spacing: -0.03em; line-height: 1;
    margin-bottom: 5px; color: inherit;
}
.stat-lbl {
    font-size: 0.7rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.4;
}
.sec-label {
    font-size: 0.7rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.09em;
    opacity: 0.4; margin: 0 0 0.65rem;
}
.answer-box {
    border: 1px solid rgba(128,128,128,0.18); border-radius: 12px;
    padding: 1.15rem 1.4rem; font-size: 0.93rem; line-height: 1.75;
    margin-bottom: 1.75rem; background: rgba(128,128,128,0.04);
}
.vid-group {
    display: flex; align-items: center; gap: 10px; margin: 1.5rem 0 0.65rem;
}
.vid-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.05em;
    padding: 3px 9px; border-radius: 6px;
    background: rgba(128,128,128,0.12); opacity: 0.8; white-space: nowrap;
}
.vid-name { font-size: 0.97rem; font-weight: 600; opacity: 0.9; }
.chunk {
    border: 1px solid rgba(128,128,128,0.15); border-radius: 10px;
    padding: 0.9rem 1.1rem; margin-bottom: 8px;
    background: rgba(128,128,128,0.03);
}
.chunk-meta {
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 0.55rem; flex-wrap: wrap;
}
.ts {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.73rem; font-weight: 500;
    padding: 2px 8px; border-radius: 5px;
    background: rgba(128,128,128,0.1); opacity: 0.75; white-space: nowrap;
}
.badge-high {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.04em;
    padding: 2px 9px; border-radius: 999px;
    background: rgba(34,197,94,0.12); color: #22c55e;
    border: 1px solid rgba(34,197,94,0.2);
}
.badge-med {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.04em;
    padding: 2px 9px; border-radius: 999px;
    background: rgba(234,179,8,0.1); color: #ca8a04;
    border: 1px solid rgba(234,179,8,0.2);
}
.badge-low {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.04em;
    padding: 2px 9px; border-radius: 999px;
    background: rgba(239,68,68,0.1); color: #ef4444;
    border: 1px solid rgba(239,68,68,0.2);
}
.score-pct {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; opacity: 0.3; margin-left: auto;
}
.chunk-text { font-size: 0.88rem; line-height: 1.65; opacity: 0.75; margin: 0; }
.rag-hr { border: none; border-top: 1px solid rgba(128,128,128,0.12); margin: 1.75rem 0; }
.sb-label {
    font-size: 0.68rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.09em;
    opacity: 0.35; margin: 1.25rem 0 0.5rem;
}
.sb-about { font-size: 0.83rem; line-height: 1.6; opacity: 0.6; }
.tech-item {
    display: flex; justify-content: space-between; align-items: center;
    padding: 6px 0; border-bottom: 1px solid rgba(128,128,128,0.1);
    font-size: 0.82rem;
}
.tech-k { opacity: 0.45; }
.tech-v { font-weight: 600; opacity: 0.85; font-size: 0.8rem; }
.rag-footer {
    text-align: center; margin-top: 3.5rem; padding-top: 1.5rem;
    border-top: 1px solid rgba(128,128,128,0.1);
    font-size: 0.75rem; opacity: 0.25; letter-spacing: 0.03em;
}

/* ── Buttons ── */
div[data-testid="stButton"] > button {
    font-size: 0.8rem !important; font-weight: 500 !important;
    padding: 0.45rem 0.7rem !important; border-radius: 8px !important;
    border: 1px solid rgba(128,128,128,0.2) !important;
    background: rgba(128,128,128,0.05) !important;
    width: 100% !important; height: auto !important;
    white-space: normal !important; text-align: left !important;
    line-height: 1.35 !important;
}
div[data-testid="stButton"] > button:hover {
    border-color: rgba(128,128,128,0.4) !important;
    background: rgba(128,128,128,0.1) !important;
}

/* ── Input box ── */
div[data-baseweb="base-input"] {
    background: rgba(128,128,128,0.04) !important;
    border: 1px solid rgba(128,128,128,0.25) !important;
    border-radius: 10px !important;
    min-height: 48px !important; height: 48px !important;
    display: flex !important; align-items: center !important;
    overflow: hidden !important;
}
div[data-baseweb="base-input"] input {
    font-size: 15px !important; height: 100% !important; width: 100% !important;
    padding: 0 14px !important; background: transparent !important;
    border: none !important; outline: none !important; box-shadow: none !important;
    line-height: normal !important;
}
div[data-baseweb="base-input"]::before,
div[data-baseweb="base-input"]::after { display: none !important; }
div[data-baseweb="base-input"]:focus-within {
    border-color: rgba(128,128,128,0.5) !important;
    box-shadow: 0 0 0 3px rgba(128,128,128,0.08) !important;
}
div[data-testid="stTextInput"] label { display: none !important; }

/* ── Metrics ── */
div[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,0.15) !important;
    border-radius: 10px !important; padding: 0.85rem 1rem !important;
    background: rgba(128,128,128,0.03) !important;
}
div[data-testid="stMetric"] label {
    font-size: 0.72rem !important; font-weight: 600 !important;
    letter-spacing: 0.06em !important; text-transform: uppercase !important;
    opacity: 0.45 !important;
}
div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.4rem !important; font-weight: 700 !important;
}

/* ── Expander ── */
div[data-testid="stExpander"] {
    border: 1px solid rgba(128,128,128,0.15) !important;
    border-radius: 10px !important; overflow: hidden !important;
}
div[data-testid="stExpander"] summary {
    display: flex !important; align-items: center !important;
    line-height: 1.4 !important; font-size: 0.88rem !important;
}

/* ── Dataframe ── */
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(128,128,128,0.15) !important;
    border-radius: 10px !important; overflow: hidden !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128,128,128,0.12) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("**RAG Video Intelligence**")
    st.markdown('<p class="sb-label">About</p>', unsafe_allow_html=True)
    st.markdown('<p class="sb-about">Semantic search over 51 MERN web development videos. Ask anything — get the exact video number, timestamp, and context.</p>', unsafe_allow_html=True)
    st.markdown('<p class="sb-label">Tech stack</p>', unsafe_allow_html=True)
    for k, v in {
        "LLM": "Gemini 2.5 Flash",
        "Embeddings": "gemini-embedding-001",
        "Dimensions": "3,072",
        "Search": "Cosine similarity",
        "Frontend": "Streamlit",
        "Backend": "Python 3.13",
    }.items():
        st.markdown(f'<div class="tech-item"><span class="tech-k">{k}</span><span class="tech-v">{v}</span></div>', unsafe_allow_html=True)
    st.markdown('<p class="sb-label">How it works</p>', unsafe_allow_html=True)
    st.markdown('<p class="sb-about">Your question is embedded using Gemini, compared against 4,693 video chunks via cosine similarity, then top matches are passed to Gemini Flash for a precise answer with timestamps.</p>', unsafe_allow_html=True)
    st.markdown('<p class="sb-label">Links</p>', unsafe_allow_html=True)
    st.markdown("[GitHub →](https://github.com/SwanandS572/RAG_Assistant_Project)  ·  [LinkedIn →](https://www.linkedin.com/in/swanand-sinnarkar-9167ab24b/)")

# ── API key ───────────────────────────────────────────────
API_KEY = None
try:
    from config import api_key
    API_KEY = api_key
except Exception:
    pass
if not API_KEY:
    try:
        API_KEY = st.secrets["api_key"]
    except Exception:
        pass
if not API_KEY:
    API_KEY = os.environ.get("api_key")
if not API_KEY:
    st.error("API key not found. Add it to config.py or Hugging Face Secrets.")
    st.stop()

client = genai.Client(api_key=API_KEY)

# ── Load embeddings ───────────────────────────────────────
@st.cache_resource
def load_embeddings():
    """Load joblib file and ensure embeddings are a clean numpy matrix."""
    raw = joblib.load("embeddings_gemini.joblib")
    # Convert embedding column to a proper 2D numpy array — avoids recursion errors
    if isinstance(raw["embedding"].iloc[0], (list, np.ndarray)):
        emb_matrix = np.vstack(raw["embedding"].values)
    else:
        emb_matrix = np.array(raw["embedding"].tolist(), dtype=np.float32)
    return raw, emb_matrix

def fmt(s):
    return f"{int(s//60):02d}:{int(s%60):02d}"

try:
    df, emb_matrix = load_embeddings()
except FileNotFoundError:
    st.error("embeddings_gemini.joblib not found. Run regenerate_embeddings_gemini.py first.")
    st.stop()
except Exception as e:
    st.error(f"Error loading embeddings: {e}")
    st.stop()

# ── Header ────────────────────────────────────────────────
st.markdown(f"""
<div class="rag-header">
    <p class="rag-title">RAG Video Intelligence</p>
    <p class="rag-subtitle">Semantic search over 51 MERN web development tutorials — exact video and timestamp every time.</p>
</div>
<div class="status-pill">
    <span class="status-dot"></span>
    System ready &nbsp;&middot;&nbsp; {len(df):,} chunks &nbsp;&middot;&nbsp; 51 videos &nbsp;&middot;&nbsp; 3,072-dim embeddings
</div>
<div class="stats-grid">
    <div class="stat-box"><div class="stat-val">51</div><div class="stat-lbl">Videos processed</div></div>
    <div class="stat-box"><div class="stat-val">{len(df):,}</div><div class="stat-lbl">Chunks indexed</div></div>
    <div class="stat-box"><div class="stat-val">3,072</div><div class="stat-lbl">Embedding dims</div></div>
</div>
""", unsafe_allow_html=True)

# ── Example buttons ───────────────────────────────────────
SHORT = ["Where is SSR taught?", "What is CSS Overflow?", "How to use useState?", "Explain JS Promises", "What is Flexbox?"]
FULL  = ["Where is SSR (Server Side Rendering) taught?", "What is CSS Overflow property?", "How to use useState in React?", "Explain JavaScript Promises", "What is Flexbox and how does it work?"]

st.markdown('<p class="sec-label">Try an example</p>', unsafe_allow_html=True)

# Initialise the input widget key before it renders
if "query_input" not in st.session_state:
    st.session_state.query_input = ""

ecols = st.columns(len(SHORT))
for i, (s, f) in enumerate(zip(SHORT, FULL)):
    with ecols[i]:
        if st.button(s, key=f"ex_{i}"):
            # Write directly into the widget key — works locally AND on HuggingFace
            st.session_state.query_input = f
            st.rerun()

# ── Search input ──────────────────────────────────────────
st.markdown('<p class="sec-label" style="margin-top:1.25rem">Your question</p>', unsafe_allow_html=True)

# No value= argument — key= owns the value via session state
query = st.text_input(
    label="search_box",
    placeholder="e.g. How do I create a responsive navbar using CSS Grid?",
    key="query_input",
    label_visibility="hidden",
)

# ── Process query ─────────────────────────────────────────
if query and query.strip():
    with st.spinner("Searching through video chunks…"):
        # Retry embedding call on 503/429
        resp = None
        for attempt in range(4):
            try:
                resp = client.models.embed_content(model="gemini-embedding-001", contents=query)
                break
            except Exception as e:
                err_str = str(e)
                if "503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    import time
                    time.sleep(2 ** attempt)
                else:
                    raise

        if resp is None:
            st.warning("⚠️ Gemini embedding service is temporarily unavailable (503). Please try again in a few seconds.")
            st.stop()

        q_emb = np.array(resp.embeddings[0].values, dtype=np.float32).reshape(1, -1)
        # Use pre-computed matrix — fast cosine similarity
        sims = cosine_similarity(emb_matrix, q_emb).flatten()
        top_idx = sims.argsort()[::-1][:10]
        top_chunks = df.iloc[top_idx].copy()
        top_chunks["score"] = sims[top_idx]

    prompt = f"""You are a teaching assistant for a MERN web development course (Sigma Web Development by CodeWithHarry).
Here are the top matching video chunks:
{top_chunks[["Video_title","Video_num","start","end","text"]].to_json(orient="records")}
Question: {query}
Give a clear, direct answer. Mention the video number and timestamp (mm:ss) where this is covered.
2-4 sentences max. Plain text only."""

    with st.spinner("Generating answer…"):
        llm_out = None
        last_error = None
        for attempt in range(4):          # up to 4 tries
            try:
                llm_out = client.models.generate_content(
                    model="gemini-2.5-flash-lite", contents=prompt
                )
                break                     # success — exit retry loop
            except Exception as e:
                last_error = e
                err_str = str(e)
                # Only retry on 503 / 429 (rate limit) — raise anything else immediately
                if "503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    import time
                    wait = 2 ** attempt   # 1s, 2s, 4s, 8s
                    time.sleep(wait)
                else:
                    raise

    st.markdown('<hr class="rag-hr">', unsafe_allow_html=True)
    st.markdown('<p class="sec-label">Answer</p>', unsafe_allow_html=True)

    if llm_out is None:
        st.warning(
            f"⚠️ Gemini is temporarily overloaded (503). "
            f"The matched video chunks below are still accurate — "
            f"check the timestamps directly. Try again in a few seconds.",
            icon="⚠️"
        )
    else:
        st.markdown(f'<div class="answer-box">{llm_out.text}</div>', unsafe_allow_html=True)
    st.markdown('<p class="sec-label">Matched video chunks</p>', unsafe_allow_html=True)

    for vid in top_chunks["Video_num"].unique():
        vc = top_chunks[top_chunks["Video_num"] == vid]
        title = vc.iloc[0]["Video_title"].replace("Sigma Web Development Course Tutorial", "").strip(" -\u2013|:")
        st.markdown(
            f'<div class="vid-group"><span class="vid-num">VIDEO {vid}</span><span class="vid-name">{title}</span></div>',
            unsafe_allow_html=True
        )
        for _, row in vc.iterrows():
            s = row["score"]
            badge = (
                '<span class="badge-high">High</span>' if s > 0.7 else
                '<span class="badge-med">Moderate</span>' if s > 0.4 else
                '<span class="badge-low">Low</span>'
            )
            preview = row["text"][:260] + ("\u2026" if len(row["text"]) > 260 else "")
            st.markdown(f"""
<div class="chunk">
    <div class="chunk-meta">
        <span class="ts">{fmt(row["start"])} \u2192 {fmt(row["end"])}</span>
        {badge}
        <span class="score-pct">{s*100:.0f}%</span>
    </div>
    <p class="chunk-text">{preview}</p>
</div>""", unsafe_allow_html=True)

    st.markdown('<hr class="rag-hr">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top match",        f"{top_chunks['score'].iloc[0]*100:.0f}%")
    c2.metric("Videos found",     int(top_chunks["Video_num"].nunique()))
    c3.metric("Chunks retrieved", len(top_chunks))
    c4.metric("Avg relevance",    f"{top_chunks['score'].mean()*100:.0f}%")

    with st.expander("Full chunk table"):
        tbl = pd.DataFrame([{
            "Video":   r["Video_num"],
            "Title":   r["Video_title"].replace("Sigma Web Development Course Tutorial","").strip(" -\u2013|:")[:38],
            "Start":   fmt(r["start"]),
            "End":     fmt(r["end"]),
            "Match":   f"{r['score']*100:.0f}%",
            "Preview": r["text"][:100] + "\u2026",
        } for _, r in top_chunks.iterrows()])
        st.dataframe(tbl, use_container_width=True, hide_index=True)

    with st.expander("Full chunk text"):
        for _, r in top_chunks.iterrows():
            t = r["Video_title"].replace("Sigma Web Development Course Tutorial","").strip(" -\u2013|:")
            st.markdown(f"**Video {r['Video_num']}: {t}** &nbsp;`{fmt(r['start'])} \u2192 {fmt(r['end'])}`&nbsp; {r['score']*100:.0f}%")
            st.markdown(r["text"])
            st.markdown('<hr class="rag-hr">', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────
st.markdown(
    '<div class="rag-footer">RAG Video Intelligence &nbsp;&middot;&nbsp; Gemini AI + Cosine Similarity &nbsp;&middot;&nbsp; Built by Swanand Sinnarkar</div>',
    unsafe_allow_html=True
)