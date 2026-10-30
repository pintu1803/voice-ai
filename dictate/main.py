from common.vad import VoiceActivityDetector
from dictate.dictate_recorder import DictateRecorder
from common.transcriber import SpeechRecognizer
from common.utils import save_wav
from common.llm import GroqLLM

import time
from common.config import settings
from scipy.io.wavfile import write


vad = VoiceActivityDetector()
recorder = DictateRecorder(vad)
recognizer = SpeechRecognizer()
llm = GroqLLM(settings.groq_api_key, settings.groq_model, settings.groq_base_url)

timestamp = time.strftime("%Y%m%d_%H%M%S")
audio_filename = f"{settings.audio_dir}/recording_{timestamp}.wav"
# speech_filename = f"{BASE_DIR}/audio/speech_{timestamp}.wav"

#Record
audio, reason = recorder.record()

if audio is None:
    print(f"No speech detected (reason: {reason})")
else:
    save_wav(audio_filename, audio)
    transcription_result = recognizer.transcribe(audio_filename)
    print("Transcript (using VAD): ", transcription_result.text)

    for token in llm.stream(transcription_result.text):
        print(token, end="", flush=True)
    print()