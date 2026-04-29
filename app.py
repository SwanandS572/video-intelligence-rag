# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import os
# from google import genai
# from sklearn.metrics.pairwise import cosine_similarity

# # Get API key from Hugging Face Secrets (NOT from config.py)
# try:
#     API_KEY = st.secrets["api_key"]  # ← Match your secret name
# except:
#     API_KEY = os.environ.get("api_key")
#     if not API_KEY:
#         st.error("❌ API key not found")
#         st.stop()

# st.set_page_config(page_title="RAG Video Assistant", page_icon="🤖")
# st.title("🤖 RAG Video Intelligence System")
# st.caption("Ask any question — get exact video number and timestamp")

# # Initialize client with API key from secrets
# client = genai.Client(api_key=API_KEY)

# @st.cache_resource
# def load_embeddings():
#     return joblib.load("embeddings_gemini.joblib")

# def format_timestamp(seconds):
#     minutes = int(seconds // 60)
#     secs = int(seconds % 60)
#     return f"{minutes}:{secs:02d}"

# # Load embeddings
# try:
#     df = load_embeddings()
#     st.success(f"✅ Loaded {len(df)} video chunks")
# except FileNotFoundError:
#     st.error("❌ embeddings_gemini.joblib not found!")
#     st.stop()
# except Exception as e:
#     st.error(f"❌ Error loading embeddings: {e}")
#     st.stop()

# query = st.text_input("Ask a question:")

# if query:
#     with st.spinner("Searching..."):
#         # Create query embedding
#         response = client.models.embed_content(
#             model="gemini-embedding-001",
#             contents=query
#         )
#         q_emb = response.embeddings[0].values
        
#         # Find similar chunks
#         similarities = cosine_similarity(np.vstack(df['embedding']), [q_emb]).flatten()
#         top_idx = similarities.argsort()[::-1][:10]
#         top_chunks = df.iloc[top_idx].copy()
#         top_chunks["similarity_score"] = similarities[top_idx]
    
#     # Display answer section
#     st.markdown("### 📖 Answer")
    
#     # Group chunks by video
#     videos = top_chunks['Video_num'].unique()
    
#     for video_num in videos:
#         video_chunks = top_chunks[top_chunks['Video_num'] == video_num]
#         video_title = video_chunks.iloc[0]['Video_title'].replace("Sigma Web Development Course Tutorial", "").strip()
        
#         st.markdown(f"**Video {video_num}: {video_title}**")
        
#         # List each chunk with its REAL timestamp
#         for _, row in video_chunks.iterrows():
#             st.markdown(f"• **{format_timestamp(row['start'])}** - {row['text'][:150]}")
        
#         st.markdown("")  # Add blank line between videos
    
#     # Optional: Use Gemini for a summary (without timestamps)
#     with st.expander("🤖 AI Summary (concept overview - no timestamps)"):
#         # Collect all relevant text
#         all_text = " ".join(top_chunks['text'].tolist())
        
#         summary_prompt = f"""Based on this course content, answer the user's question in 2-3 sentences.

# User question: {query}

# Course content: {all_text[:2000]}

# Provide a brief summary. DO NOT include any timestamps or video numbers. Just explain the concept."""

#         try:
#             summary = client.models.generate_content(
#                 model="gemini-2.5-flash-lite",
#                 contents=summary_prompt
#             )
#             st.write(summary.text)
#         except:
#             st.write("Summary not available")
    
#     # Show data table
#     st.markdown("---")
#     st.markdown("### 📊 All Matched Chunks")
    
#     display_data = []
#     for _, row in top_chunks.iterrows():
#         clean_title = row['Video_title'].replace("Sigma Web Development Course Tutorial", "").strip()
#         display_data.append({
#             "Video": row['Video_num'],
#             "Title": clean_title[:40],
#             "Time": format_timestamp(row['start']),
#             "Text": row['text'][:100] + "..." if len(row['text']) > 100 else row['text'],
#             "Relevance": f"{row['similarity_score']*100:.1f}%"
#         })
    
#     display_df = pd.DataFrame(display_data)
#     st.dataframe(display_df, width='stretch', hide_index=True)
    
#     # Detailed expander
#     with st.expander("🔍 View full text of all chunks"):
#         for _, row in top_chunks.iterrows():
#             st.markdown(f"**Video {row['Video_num']}: {row['Video_title']}**")
#             st.markdown(f"⏱️ **Timestamp:** {format_timestamp(row['start'])} - {format_timestamp(row['end'])}")
#             st.markdown(f"📝 **Text:** {row['text']}")
#             st.markdown(f"📊 **Relevance:** {row['similarity_score']*100:.1f}%")
#             st.markdown("---")


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from google import genai
from sklearn.metrics.pairwise import cosine_similarity

# Get API key - check local config.py FIRST, then HF secrets
API_KEY = None

# First try local config.py (for local development)
try:
    from config import api_key
    API_KEY = api_key
    st.success("✅ Using local API key from config.py")
except:
    pass

# If no local key, try Hugging Face secrets (for cloud deployment)
if not API_KEY:
    try:
        API_KEY = st.secrets["api_key"]
        st.success("✅ Using API key from Hugging Face Secrets")
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

st.set_page_config(page_title="RAG Video Assistant", page_icon="🤖")
st.title("🤖 RAG Video Intelligence System")
st.caption("Ask any question — get exact video number and timestamp")

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

query = st.text_input("Ask a question about web development:")

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