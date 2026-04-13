import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity 
import joblib
import requests
from google import genai
from config import api_key  

def create_embeddings(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    embedding = r.json()["embeddings"]
    return embedding


def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })
    return r.json()['response'] 
    # print(response)

client = genai.Client(api_key=api_key)

def inference_geminiapi(prompt):
    response = client.models.generate_content(
        # model="gemini-2.5-flash",   # BEST for accuracy
        model="gemini-2.5-flash-lite",
        contents=prompt,
    )
    return response.text

df = joblib.load('embeddings.joblib')
incoming_query = input("Ask a Question: ")
question_embedding = create_embeddings([incoming_query])[0]
# print(question_embedding)

similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# print(similarities) 
top_results = 10
max_index = similarities.argsort()[::-1][0:top_results]
print(max_index)
new_df = df.loc[max_index]
# print(new_df[["title", "number", "text"]])
# print(new_df[["Video_title", "Video_num", "text"]])  # ✅ Correct names

# for index,item in new_df.iterrows():
#     print(index,item["Video_title"],item["Video_num"],item["text"],item["start"],item["end"])

# prompt = f'''I am teaching web development in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

# {new_df[["Video_title", "Video_num", "start", "end", "text"]].to_json(orient="records")}
# ---
# {incoming_query}
# User asked this question related to the video chunks, you have to answer where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
# '''

prompt = f'''I am teaching web development in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

Question: {incoming_query}
---

{new_df[["Video_title", "Video_num", "start", "end", "text"]].to_json(orient="records")}

Answer where and how much content is taught in which video (video number and timestamp). Guide the user to go to that particular video. If the user asks an unrelated question, politely tell them you can only answer questions related to the Sigma Web Development course.

IMPORTANT: Use plain text only. Do NOT use markdown formatting like **bold** or *italic*. Write in normal sentences.'''

with open("prompt.txt","w") as f:
    f.write(prompt)

# response = inference(prompt)
# print(response)

response = inference_geminiapi(prompt)
print(response)

# with open("response.txt","w",encoding="latin") as f:
#     f.write(response)

with open("response.txt", "w", encoding="utf-8") as f:
    f.write(response)