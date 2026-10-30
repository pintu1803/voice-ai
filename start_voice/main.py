from common.vad import VoiceActivityDetector
from common.transcriber import SpeechRecognizer
from start_voice.recorder import Recorder

import time

vad = VoiceActivityDetector()
recorder = Recorder(vad)
recognizer = SpeechRecognizer()


timestamp = time.strftime("%Y%m%d_%H%M%S")
audio_filename = f"./audio/recording_{timestamp}.wav"
speech_filename = f"./audio/speech_{timestamp}.wav"

#Record
recorder.record(audio_filename)
#Transcribe
# transcription_result = recognizer.transcribe(speech_filename)
# print("Transcript (using VAD): ", transcription_result.text)

# transcription_result = recognizer.transcribe(audio_filename)
# print("Transcript (without VAD) : ", transcription_result.text)



