import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity 
import joblib
import requests

def create_embeddings(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()['embeddings']
    return embedding

df = joblib.load('embeddings.joblib')
incoming_query = input("Ask a Question: ")
question_embedding = create_embeddings([incoming_query])[0]
# print(question_embedding)

similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# print(similarities) 
top_results = 3
max_index = similarities.argsort()[::-1][0:top_results]
print(max_index)
new_df = df.loc[max_index]
# print(new_df[["title", "number", "text"]])
# print(new_df[["Video_title", "Video_num", "text"]])  # ✅ Correct names

for index,item in new_df.iterrows():
    print(index,item["Video_title"],item["Video_num"],item["text"],item["start"],item["end"])
