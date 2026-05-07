"""
Regenerate embeddings using Gemini API - FULLY WORKING VERSION
"""

import pandas as pd
import joblib
from google import genai
from config import api_key
import time

print("🔄 Regenerating embeddings using Gemini API...")

# Initialize client
client = genai.Client(api_key=api_key)

# List of possible working model names for google-genai
MODELS_TO_TRY = [
    "models/embedding-001",
    "embedding-001", 
    "text-embedding-001",
    "gemini-embedding-001",
]

def find_working_model():
    """Find which embedding model works"""
    for model in MODELS_TO_TRY:
        try:
            print(f"Testing model: {model}...", end=" ")
            response = client.models.embed_content(
                model=model,
                contents="test"
            )
            print(f"✅ WORKING!")
            return model
        except Exception as e:
            print(f"❌ Failed")
    return None

def create_embedding(text, model_name):
    """Create a single embedding with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.embed_content(
                model=model_name,
                contents=text
            )
            return response.embeddings[0].values
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = 35
                print(f"\n⚠️ Rate limit! Waiting {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"\n❌ Error: {str(e)[:100]}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(5)
    return None

# Step 1: Find working model
print("\n🔍 Finding working embedding model...")
working_model = find_working_model()

if not working_model:
    print("\n❌ No working embedding model found!")
    print("💡 Alternative: Use Sentence Transformers instead:")
    print("   pip install sentence-transformers")
    exit(1)

print(f"\n✅ Using model: {working_model}")

# Step 2: Load your data
print("\n📂 Loading original data...")
try:
    old_df = joblib.load('embeddings.joblib')
    # Remove old embeddings if they exist
    df = old_df.drop('embedding', axis=1) if 'embedding' in old_df.columns else old_df
    print(f"✅ Loaded {len(df)} chunks")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit(1)

# Step 3: Generate embeddings
print(f"\n📝 Generating embeddings for {len(df)} chunks...")
print("⏱️ This will take time due to API rate limits (100 requests/minute)")
print("💡 The script will automatically wait when rate limit is hit\n")

embeddings = []
total = len(df)
successful = 0
failed = 0

for idx, row in df.iterrows():
    text = row['text']
    print(f"[{idx+1}/{total}] Processing...", end=" ")
    
    embedding = create_embedding(text, working_model)
    if embedding is not None:
        embeddings.append(embedding)
        successful += 1
        print("✅")
    else:
        embeddings.append(None)
        failed += 1
        print("❌")
    
    # Small delay to avoid hitting rate limits too quickly
    time.sleep(0.5)

# Step 4: Add embeddings and save
df['embedding'] = embeddings

# Remove failed ones
if failed > 0:
    print(f"\n⚠️ {failed} chunks failed, removing them...")
    df = df.dropna(subset=['embedding'])

# Save
output_file = 'embeddings_gemini.joblib'
joblib.dump(df, output_file)

print(f"\n{'='*50}")
print(f"✅ SUCCESS! Saved to {output_file}")
print(f"   Total successful chunks: {len(df)}/{total}")
if len(df) > 0:
    print(f"   Embedding dimension: {len(df['embedding'].iloc[0])}")
print(f"{'='*50}")