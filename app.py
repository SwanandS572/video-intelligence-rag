import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from google import genai
from sklearn.metrics.pairwise import cosine_similarity

# ⚠️ MUST BE FIRST - before any other st.xxx commands
st.set_page_config(page_title="RAG Video Assistant", page_icon="🤖")

# Now you can do other Streamlit commands
st.title("🤖 RAG Video Intelligence System")
st.caption("Ask any question — get exact video number and timestamp")

# Get API key - check local config.py FIRST, then HF secrets
API_KEY = None

# First try local config.py (for local development)
try:
    from config import api_key
    API_KEY = api_key
except:
    pass

# If no local key, try Hugging Face secrets (for cloud deployment)
if not API_KEY:
    try:
        API_KEY = st.secrets["api_key"]
        # st.success("✅ Using API key from Hugging Face Secrets")  # ← Optional: remove or keep
    except:
        pass

# If still no key, try environment variable
if not API_KEY:
    API_KEY = os.environ.get("api_key")

# If no key found, show error
if not API_KEY:
    st.error("❌ API key not found!")
    st.info("Please add your API key in config.py or Hugging Face Secrets")
    st.stop()

# Initialize client with API key
client = genai.Client(api_key=API_KEY)

@st.cache_resource
def load_embeddings():
    return joblib.load("embeddings_gemini.joblib")

def format_timestamp(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"

# Load embeddings
try:
    df = load_embeddings()
    st.success(f"✅ Loaded {len(df)} video chunks")
except FileNotFoundError:
    st.error("❌ embeddings_gemini.joblib not found!")
    st.info("Run: python regenerate_embeddings_gemini.py to create embeddings")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading embeddings: {e}")
    st.stop()

# Example questions placeholder
example_questions = [
    "Where is SSR (Server Side Rendering) taught?",
    "What is CSS Overflow property?",
    "How to use useState in React?",
    "Explain JavaScript Promises",
    "What is Flexbox and how does it work?"
]

# Display example questions as buttons
st.markdown("### 💡 Example Questions")
cols = st.columns(len(example_questions))

for idx, example in enumerate(example_questions):
    with cols[idx]:
        if st.button(example, key=f"ex_{idx}"):
            st.session_state.query = example

# Use session state to persist query
if "query" not in st.session_state:
    st.session_state.query = ""

query = st.text_input("Ask a question about web development (MERN):", value=st.session_state.query)

if query:
    with st.spinner("Searching through videos..."):
        # Create query embedding
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=query
        )
        q_emb = response.embeddings[0].values
        
        # Find similar chunks
        similarities = cosine_similarity(np.vstack(df['embedding']), [q_emb]).flatten()
        top_idx = similarities.argsort()[::-1][:10]
        top_chunks = df.iloc[top_idx].copy()
        top_chunks["similarity_score"] = similarities[top_idx]
    
    # Display answer section
    st.markdown("### 📖 Answer")
    
    # Group chunks by video
    for video_num in top_chunks['Video_num'].unique():
        video_chunks = top_chunks[top_chunks['Video_num'] == video_num]
        video_title = video_chunks.iloc[0]['Video_title'].replace("Sigma Web Development Course Tutorial", "").strip()
        
        st.markdown(f"**Video {video_num}: {video_title}**")
        
        for _, row in video_chunks.iterrows():
            st.markdown(f"• **{format_timestamp(row['start'])}** - {row['text'][:150]}")
        
        st.markdown("")
    
    # Show data table
    st.markdown("---")
    st.markdown("### 📊 All Matched Chunks")
    
    display_data = []
    for _, row in top_chunks.iterrows():
        clean_title = row['Video_title'].replace("Sigma Web Development Course Tutorial", "").strip()
        display_data.append({
            "Video": row['Video_num'],
            "Title": clean_title[:40],
            "Time": format_timestamp(row['start']),
            "Text": row['text'][:100] + "..." if len(row['text']) > 100 else row['text'],
            "Relevance": f"{row['similarity_score']*100:.1f}%"
        })
    
    display_df = pd.DataFrame(display_data)
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Detailed expander
    with st.expander("🔍 View full text of all chunks"):
        for _, row in top_chunks.iterrows():
            st.markdown(f"**Video {row['Video_num']}: {row['Video_title']}**")
            st.markdown(f"⏱️ **Timestamp:** {format_timestamp(row['start'])} - {format_timestamp(row['end'])}")
            st.markdown(f"📝 **Text:** {row['text']}")
            st.markdown(f"📊 **Relevance:** {row['similarity_score']*100:.1f}%")
            st.markdown("---")