import json
import os
import pandas as pd
import joblib
from google import genai
from config import api_key
import time

print("Generating embeddings using Gemini API...")

client = genai.Client(api_key=api_key)

MODELS_TO_TRY = [
    "gemini-embedding-001",
    "gemini-embedding-2-preview",
]

def find_working_model():
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

print("\n🔍 Finding working embedding model...")
working_model = find_working_model()

if not working_model:
    print("\n❌ No working embedding model found!")
    exit(1)

print(f"\n✅ Using model: {working_model}")

print("\n📂 Loading data from newjsons/ folder...")
all_chunks = []
newjsons_folder = "newjsons"

if not os.path.exists(newjsons_folder):
    print(f"❌ Folder '{newjsons_folder}' not found!")
    exit(1)

json_files = [f for f in os.listdir(newjsons_folder) if f.endswith('.json')]
print(f"Found {len(json_files)} JSON files in newjsons/")

for json_file in json_files:
    with open(os.path.join(newjsons_folder, json_file), 'r', encoding='utf-8') as f:
        data = json.load(f)
        for chunk in data['chunks']:
            all_chunks.append(chunk)

df = pd.DataFrame(all_chunks)
total_chunks = len(df)
print(f"✅ Loaded {total_chunks} merged chunks from {len(json_files)} videos")

output_file = 'embeddings_gemini.joblib'
existing_embeddings = []
start_idx = 0
processed_texts = set()

if os.path.exists(output_file):
    print(f"\n📂 Found existing embeddings file. Checking progress...")
    try:
        existing_df = joblib.load(output_file)
        existing_embeddings = existing_df['embedding'].tolist()
        processed_texts = set(existing_df['text'].tolist())
        start_idx = len(existing_df)
        print(f"   ✅ Already processed: {start_idx} chunks")
        df = df[~df['text'].isin(processed_texts)]
        print(f"   📌 Remaining to process: {len(df)} chunks")
        if len(df) == 0:
            print("\n🎉 All chunks already processed!")
            exit(0)
    except Exception as e:
        print(f"   ⚠️ Could not read existing file: {e}")
        existing_embeddings = []
        processed_texts = set()
        start_idx = 0
else:
    print(f"\n📂 No existing embeddings file found. Starting fresh...")
    existing_embeddings = []
    processed_texts = set()
    start_idx = 0

print(f"\n📝 Generating embeddings for {len(df)} remaining chunks...")
embeddings = list(existing_embeddings)
total = total_chunks
successful = start_idx
failed = 0
batch_counter = 0

for idx, row in df.iterrows():
    actual_index = start_idx + idx + 1
    text = row['text']
    print(f"[{actual_index}/{total}] Processing...", end=" ")
    
    embedding = create_embedding(text, working_model)
    if embedding is not None:
        embeddings.append(embedding)
        successful += 1
        print("✅")
    else:
        embeddings.append(None)
        failed += 1
        print("❌")
    
    batch_counter += 1
    if batch_counter >= 50:
        batch_counter = 0
        print(f"\n💾 Saving checkpoint... (Progress: {successful}/{total})")
        temp_df = pd.DataFrame(all_chunks[:successful])
        temp_df['embedding'] = embeddings[:successful]
        temp_df = temp_df.dropna(subset=['embedding'])
        joblib.dump(temp_df, output_file)
        print(f"   Checkpoint saved to {output_file}\n")
    
    time.sleep(0.5)

print(f"\n💾 Saving final embeddings...")
final_df = pd.DataFrame(all_chunks[:successful])
final_df['embedding'] = embeddings[:successful]
final_df = final_df.dropna(subset=['embedding'])
joblib.dump(final_df, output_file)

print(f"\n{'='*50}")
print(f"✅ SUCCESS! Saved to {output_file}")
print(f"   Total successful chunks: {len(final_df)}/{total}")
if len(final_df) > 0:
    print(f"   Embedding dimension: {len(final_df['embedding'].iloc[0])}")
print(f"{'='*50}")