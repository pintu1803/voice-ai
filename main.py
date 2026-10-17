from recorder import record_audio
from transcriber import SpeechRecognizer
import time

SAMPLE_RATE = 16000
DURATION = 15

recognizer = SpeechRecognizer()

while(True):
    cmd = input("Record more : ")
    if cmd in ["f", "n", "N",  "F", "false", "FALSE", "no", "No", "NO"]:
        break

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"./audio/recording_{timestamp}.wav"

    record_audio(filename, DURATION, SAMPLE_RATE)
    transcript = recognizer.transcribe(filename)
    print("transcript : ", transcript)