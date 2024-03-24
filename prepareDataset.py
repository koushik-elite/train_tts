import os, gc
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import pandas as pd, numpy as np
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
import pickle
from pydub import AudioSegment
import re
from sklearn.model_selection import train_test_split
import glob
from pathlib import Path

# path = "/home/koushik/sample/"
tsv_final_list = []
for name in glob.glob('/home/koushik/sample/*.tsv'):
    print(name)
    p = Path(name)
    # print(p.stem)
    tsv_data = pd.read_csv(name, names=['start','end','text'], skiprows=1, sep='\t')
    tsv_data["filename"] = p.stem + ".wav"
    tsv_list = tsv_data.values.tolist()
    tsv_final_list.extend(tsv_list)

# print(tsv_final_list)
train_df = pd.DataFrame(tsv_final_list, columns=['start','end','text','filename'])
# output_df.to_csv("/mnt/d/Data/batman/final_metadata.tsv", sep='\t', index=False)

all_data = []
index = 0
for _, row in tqdm(train_df.iterrows()):
    sound_file = AudioSegment.from_wav("/mnt/d/Data/batman/brave_no_noice/" + row.filename)
    start = int(row.start)
    end = int(row.end)
    audio = sound_file[start:end]
    audio.export("/mnt/d/Data/batman/train/wavs/audio_{0}.wav".format(index),format = "wav")
    res = row.text
    all_data.append({
        "audio": "audio_{0}".format(index),
        "raw": res,
        "text": res,
    })
    index += 1

output_df = pd.DataFrame(all_data)
output_df.to_csv("/mnt/d/Data/batman/train/metadata.txt", sep='|', header=False, index=False)