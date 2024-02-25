import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/en/ljspeech/glow-tts").to(device)

# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
# wav = tts.tts(text="Hello world!", speaker_wav="audio.wav")
# Text to speech to a file
# tts.tts_to_file(text="Hail, noble cousin. Now, let's get you from that miserable place. I'll see that all of us upon Olympus do our part, beginning here with me.", speaker_wav="/mnt/d/Data/Athena_voice/audio/chunk23.wav", file_path="output.wav")
tts.tts_to_file(text="my pussy and anus. You ass hole!. My Vagina and buttock. I will suck your penis", file_path="audio.wav")