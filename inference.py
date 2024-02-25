from TTS.api import TTS

from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.models.vits import Vits
from TTS.tts.models.glow_tts import GlowTTS
from TTS.config import load_config
from TTS.utils.synthesizer import Synthesizer

# config = GlowTTSConfig()
# model = GlowTTS(config)

config = load_config("glow-tts-finetune-February-24-2024_06+03PM-a4ba385/config.json")
model = GlowTTS.init_from_config(config)
model.load_checkpoint(config, 'glow-tts-finetune-February-24-2024_06+03PM-a4ba385/best_model_343945.pth', eval=True)

speakers_file_path = None
language_ids_file_path = None
vocoder_path = None
vocoder_config_path = None
encoder_path = None
encoder_config_path = None
cuda = True

synthesizer = Synthesizer(
    "glow-tts-finetune-February-24-2024_06+03PM-a4ba385/best_model_343945.pth", 
    "glow-tts-finetune-February-24-2024_06+03PM-a4ba385/config.json", 
    speakers_file_path, 
    language_ids_file_path, 
    vocoder_path, 
    vocoder_config_path, 
    encoder_path, 
    encoder_config_path, 
    cuda
)

speaker_idx = None
language_idx = None
speaker_wav = None
reference_wav = None
style_wav = None
style_text = None
reference_speaker_name = None
wav = synthesizer.tts(
	"Hail, noble cousin. Now, let's get you from that miserable place. I'll see that all of us upon Olympus do our part, beginning here with me",
	speaker_idx,
	language_idx,
	speaker_wav,
	reference_wav,
	style_wav,
	style_text,
	reference_speaker_name
)
synthesizer.save_wav(wav, "output.wav")