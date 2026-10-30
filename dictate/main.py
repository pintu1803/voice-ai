from common.vad import VoiceActivityDetector
from dictate.dictate_recorder import DictateRecorder
from common.transcriber import SpeechRecognizer
from common.utils import save_wav

import time
from config import (SAMPLE_RATE, DURATION, BASE_DIR)
from scipy.io.wavfile import write


vad = VoiceActivityDetector()
recorder = DictateRecorder(vad)
recognizer = SpeechRecognizer()


timestamp = time.strftime("%Y%m%d_%H%M%S")
audio_filename = f"{BASE_DIR}/audio/recording_{timestamp}.wav"
# speech_filename = f"{BASE_DIR}/audio/speech_{timestamp}.wav"

#Record
audio, reason = recorder.record()

if audio is None:
    print(f"No speech detected (reason: {reason})")
else:
    save_wav(audio_filename, audio)
    transcription_result = recognizer.transcribe(audio_filename)
    print("Transcript (using VAD): ", transcription_result.text)