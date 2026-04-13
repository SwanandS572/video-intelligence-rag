import os 
import math 
import json

n=5
for fn in os.listdir("jsons"): # fn=file_name
    if fn.endswith(".json"):
        fp = os.path.join("jsons",fn) # fp=file_path
        with open(fp,"r",encoding="utf-8") as f:
            data = json.load(f)
            new_chunks = []
            num_chunks = len(data['chunks']) # if total chunks are = 256
            num_groups = math.ceil(num_chunks/n) # ceil(256/5) = ceil(51.2) = 52 groups 

            for i in range(num_groups): # start from i=0 to i=52 groups
                start_idx = i*n
                end_idx = min((i+1)*n, num_chunks)

                chunk_group = data['chunks'][start_idx:end_idx]

                new_chunks.append({
                    "Video_num": data['chunks'][0]['Video_num'],
                    "Video_title": chunk_group[0]['Video_title'],
                    "start": chunk_group[0]['start'],
                    "end": chunk_group[-1]['end'],
                    "text": " ".join(c['text'] for c in chunk_group)
                })
            
            os.makedirs("newjsons", exist_ok=True)
            with open(os.path.join("newjsons", fn), "w", encoding="utf-8") as json_file: # auto close file without f.close()
                json.dump({"chunks":new_chunks, "text": data['text']}, json_file, indent=4)