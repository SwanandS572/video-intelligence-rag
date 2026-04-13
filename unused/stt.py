import os
import json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import whisper

model = whisper.load_model("base")
result = model.transcribe(audio="audios/1_Basic Structure of an HTML Website  Sigma Web Development Course Tutorial.mp3", language="hi", task="translate", word_timestamps=False)

# print(result["segments"])
chunks=[]
for segment in result["segments"]:
    chunks.append({"start": segment["start"], "end": segment["end"], "text": segment["text"]})

print(chunks)

with open("output.json", "w") as f:
    json.dump(chunks, f)