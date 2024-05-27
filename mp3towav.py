import os
from os import path
from pydub import AudioSegment
from tqdm import tqdm

DATASET_PATH = "/home/heinecantor/Desktop/dataset/dataset/audio/"
OUTPUT_PATH = "/home/heinecantor/Desktop/dataset/dataset/audio/conversion/"

os.makedirs(OUTPUT_PATH, exist_ok=True)

for file in tqdm(os.listdir(DATASET_PATH)):
    if file.endswith(".mp3"):
        src = path.join(DATASET_PATH, file)
        dst = path.join(OUTPUT_PATH, file.replace(".mp3", ".wav"))
        sound = AudioSegment.from_mp3(src)
        sound.export(dst, format="wav")