import requests
import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

def create_embeddings(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()['embeddings']
    return embedding

# a=create_embeddings("Cat sat on the mat")
# print(a)

jsons = os.listdir("newjsons") # List all the jsons
# print(jsons)
mydicts=[]
chunk_id = 0

for j in jsons:
    with open(f"newjsons/{j}") as f:
        content = json.load(f)
    print(f"Creating Embeddings for {j} file")
    embeddings = create_embeddings([c['text'] for c in content['chunks']])
    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        mydicts.append(chunk)
    #     if(i==3):
    #         break
    # break

# print(mydicts)

df = pd.DataFrame.from_records(mydicts)
# print(df)
joblib.dump(df, 'embeddings.joblib')


