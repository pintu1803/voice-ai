from common.vad import VoiceActivityDetector
import sounddevice as sd
import torch

from collections import deque
from scipy.io.wavfile import write
from start_voice.recorder_state import RecorderState
from common.utils import (save_wav)

from common.config import settings

SAMPLE_RATE = settings.sample_rate
SILENCE_TIMEOUT = settings.silence_timeout 
PRE_ROLL_CHUNKS = settings.pre_roll_chunks
BLOCK_SIZE = settings.block_size

class Recorder:

    def __init__(self, vad : VoiceActivityDetector):
        self.vad = vad
        self.state = RecorderState.WAITING
        self.pre_roll = deque(maxlen=PRE_ROLL_CHUNKS)
        self.audio_buffer = []
        self.silence_duration = 0


    def convert_chunk(self, chunk):
        chunk = chunk.squeeze()
        chunk = torch.tensor(chunk, dtype=torch.float32)
        chunk /= 32768.0
        return chunk

    
    def record(self, filename):
        print("Listening...")
        self.reset()

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=BLOCK_SIZE
        ) as stream:

            while True:

                chunk, overflow = stream.read(BLOCK_SIZE)
                chunk = self.convert_chunk(chunk)
                event = self.vad.process(chunk)

                audio = self.handle_chunk(chunk, event)

                if audio is not None:

                    save_wav(filename, audio)
                    print("Saved:", filename)
                    # 1. records one monologue
                    break
                    # 2. records multiple monologues
                    # self.reset()


    def handle_chunk(self, chunk, event):

        if self.state == RecorderState.WAITING:
            self.pre_roll.append(chunk)

            if event.speech_started:
                print("Speech started")
                self.state = RecorderState.RECORDING
                self.audio_buffer.extend(self.pre_roll)
                self.pre_roll.clear()

        elif self.state == RecorderState.RECORDING:
            self.audio_buffer.append(chunk)

            if event.speech_ended:
                print("Speech ended")
                self.state = RecorderState.ENDING
                self.silence_duration = 0

        elif self.state == RecorderState.ENDING:
            self.audio_buffer.append(chunk)
            self.silence_duration += (BLOCK_SIZE / SAMPLE_RATE * 1000)

            if event.speech_started:
                print("Speech resumed")
                self.state = RecorderState.RECORDING
                self.silence_duration = 0

            elif self.silence_duration >= SILENCE_TIMEOUT:
                print("Recording finished")

                return torch.cat(self.audio_buffer)

        return None


    def reset(self):
        self.state = RecorderState.WAITING
        self.pre_roll.clear()
        self.audio_buffer = []
        self.silence_duration = 0