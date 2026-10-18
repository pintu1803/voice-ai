from recorder import record_audio
from transcriber import SpeechRecognizer
from vad import VoiceActivityDetector
import time
from config import (SAMPLE_RATE, DURATION)

recognizer = SpeechRecognizer()
vad = VoiceActivityDetector()


while(True):
    cmd = input("Record more : ")
    if cmd in ["f", "n", "N",  "F", "false", "FALSE", "no", "No", "NO"]:
        break

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    audio_filename = f"./audio/recording_{timestamp}.wav"
    speech_filename = f"./audio/speech_{timestamp}.wav"

    #Record
    record_audio(audio_filename, DURATION, SAMPLE_RATE)
    #Remove silence
    vad.remove_silence(audio_filename, speech_filename)
    #Transcribe
    transcription_result = recognizer.transcribe(speech_filename)
    print("Transcript (using VAD): ", transcription_result.text)

    transcription_result = recognizer.transcribe(audio_filename)
    print("Transcript (without VAD) : ", transcription_result.text)