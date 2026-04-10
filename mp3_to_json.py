import os
import json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import whisper

model = whisper.load_model("base")

audio = os.listdir("audios")

for x in audio:
    num = x.split("_")[0]
    title = x.split("_")[1][:-4]
    print(num, title)
    result = model.transcribe(audio=f"audios/{x}", language="hi", task="translate", word_timestamps=False)

    chunks=[]
    for segment in result["segments"]:
        chunks.append({"Video_num": num ,"Video_title":title, "start": segment["start"], "end": segment["end"], "text": segment["text"]})

    chunks_with_metadata = {"chunks": chunks,"text": result["text"]}

    with open(f"jsons/{x}.json", "w") as f:
        json.dump(chunks_with_metadata, f)