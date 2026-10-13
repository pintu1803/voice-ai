import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
DURATION = 5

print("Speak...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="int16"
)

sd.wait()

write("audio.wav", SAMPLE_RATE, audio)

print("Transcribing...")

model = WhisperModel(
    "./models/faster-whisper-base",
    device="cpu",
    compute_type="int8"
)

segments, info = model.transcribe("audio.wav")

print()

print("Detected language:", info.language)

print()

for segment in segments:
    print(segment.text)