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

sound_file = AudioSegment.from_mp3("sample/audio/Hades_voice_line_Athena.mp3")
train = pd.read_csv('sample/audio/Hades_voice_line_Athena.tsv', names=['start',	'end',	'text'], skiprows=1, sep='\t')
train["text"] = train['text'].str.replace('[^\w\s]','')
print(train.head())
index = 0
all_data = []
for _, row in tqdm(train.iterrows()):
    start = int(row.start)
    end = int(row.end)
    audio = sound_file[start:end]
    mono_audio = audio.split_to_mono()
    if len(mono_audio) > 1:
        mono_audio = mono_audio[0]
    mono_audio = mono_audio.set_frame_rate(22050)
    mono_audio.export(
        "/mnt/d/Data/Athena_voice/wavs/audio{0}.wav".format(index),
        bitrate = "16k", format = "wav"
    )

    # res = re.sub(r'[^\w\s]', '', row.text)
    res = row.text
    all_data.append({
        "audio": "audio{0}".format(index),
        "raw": res,
        "text": res,
    })
    index += 1
    # print(start, )


output_df = pd.DataFrame(all_data)
output_len = output_df.shape[0]

valid_len = int(output_len * 0.2)
train_len = output_len - valid_len

train_df = output_df[:train_len]
valid_df = output_df[train_len:]

output_df.to_csv("/mnt/d/Data/Athena_voice/metadata.txt", sep='|', header=False, index=False)
train_df.to_csv("/mnt/d/Data/Athena_voice/train_audio.txt", sep='|', header=False, index=False)
valid_df.to_csv("/mnt/d/Data/Athena_voice/valid_audio.txt", sep='|', header=False, index=False)