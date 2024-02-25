import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=True).to(device)
tts.voice_conversion_to_file(source_wav="audio.wav", target_wav="/mnt/d/Data/Athena_voice/wavs/audio0.wav", file_path="output.wav")
