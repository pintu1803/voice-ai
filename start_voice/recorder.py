from common.vad import VoiceActivityDetector
import sounddevice as sd
import torch

from collections import deque
from scipy.io.wavfile import write
from start_voice.recorder_state import RecorderState
from common.utils import (save_wav)

from common.config import settings

SAMPLE_RATE = settings.sample_rate
SILENCE_TIMEOUT = settings.end_of_speech_silence_ms 
PRE_ROLL_CHUNKS = settings.pre_roll_chunks
BLOCK_SIZE = settings.block_size
chunk_ms = (BLOCK_SIZE / SAMPLE_RATE * 1000)

class Recorder:

    def __init__(self, vad : VoiceActivityDetector):
        self.vad = vad
        self.state = RecorderState.WAITING
        self.pre_roll = deque(maxlen=PRE_ROLL_CHUNKS)
        self.audio_buffer = []
        self.silence_duration = 0
        # self.stop_event = False


    def convert_chunk(self, chunk):
        chunk = chunk.squeeze()
        chunk = torch.tensor(chunk, dtype=torch.float32)
        chunk /= 32768.0
        return chunk

    
    def listen(self, stop_event, voice_idle_timeout_ms):
        self.reset()

        idle_ms = 0

        print("Listening...")

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=BLOCK_SIZE
        ) as stream:

            while not stop_event.is_set():

                raw, overflow = stream.read(BLOCK_SIZE)
                chunk = self.convert_chunk(raw) 
                event = self.vad.process(chunk)

                audio = self.handle_chunk(chunk, event)

                if audio is not None:
                    return audio, "utterance"
                    # save_wav(filename, audio)
                    # print("Saved:", filename)
                    # break
                
                if self.state == RecorderState.WAITING:
                    idle_ms += chunk_ms
                    if idle_ms >= settings.voice_idle_timeout_ms:
                        return None, "idle_timeout"

        return None, "stopped"

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
            self.silence_duration += chunk_ms

            if event.speech_started:
                print("Speech resumed")
                self.state = RecorderState.RECORDING
                self.silence_duration = 0

            elif self.silence_duration >= SILENCE_TIMEOUT:
                print("Recording finished")

                return torch.cat(self.audio_buffer)

        # recording not finished # voice command still building
        return None


    def reset(self):
        self.state = RecorderState.WAITING
        self.pre_roll.clear()
        self.audio_buffer = []
        self.silence_duration = 0