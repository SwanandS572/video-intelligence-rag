import os
import subprocess

files = os.listdir("videos")
for file in files:
    tno = file.split(" -")[0].split("#")[1]
    file_name = file.split(" #")[0]
    print(tno, file_name)
    subprocess.run(["ffmpeg", "-i", f"videos/{file}", f"audios/{tno}_{file_name}.mp3"])