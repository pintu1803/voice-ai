import sounddevice as sd
from scipy.io.wavfile import write
from time import time

def record_audio(filename: str, DURATION: int = 5, SAMPLE_RATE: int = 16000):

    print("Speak...")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16"
    )

    sd.wait()
    write(filename, SAMPLE_RATE, audio)