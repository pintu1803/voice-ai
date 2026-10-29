import threading
import sounddevice as sd
import torch

from config import (SAMPLE_RATE, BLOCK_SIZE, DICTATE_IDLE_TIMEOUT_MS)

TAIL_CHUNKS = 15   # keep a little silence after the last speech (~0.5 s at 512 samples/16 kHz)


class DictateRecorder:

    def __init__(self, vad):
        self.vad = vad
        self.stop_event = False
        self.result = {}

    def stop(self):
        """Call this from the UI/hotkey thread to end the dictation."""
        self.stop_event = True

    def convert_chunk(self, chunk):
        chunk = torch.tensor(chunk.squeeze(), dtype=torch.float32)
        chunk /= 32768.0
        return chunk

    def record_main(self):
        """
        Blocks until stop() is called or the idle timeout fires.
        Returns (audio_tensor_or_None, reason)
        reason is "manual" or "idle_timeout"
        """
        self.stop_event = False
        # self.vad.reset()          # clear VAD hidden state, adapt to your wrapper

        chunks = []
        speaking = False
        heard_speech = False
        idle_ms = 0
        last_speech_idx = 0       # number of chunks up to the last speech activity
        chunk_ms = BLOCK_SIZE / SAMPLE_RATE * 1000
        reason = "manual"

        print("Dictating... (stop manually, or auto-stops after 60 s of silence)")

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=BLOCK_SIZE
        ) as stream:

            while not self.stop_event:
                raw, overflow = stream.read(BLOCK_SIZE)
                chunk = self.convert_chunk(raw)
                event = self.vad.process(chunk)
                chunks.append(chunk)

                if event.speech_started:
                    speaking = True
                    heard_speech = True
                if event.speech_ended:
                    speaking = False
                    last_speech_idx = len(chunks)

                if speaking:
                    idle_ms = 0
                    last_speech_idx = len(chunks)
                else:
                    idle_ms += chunk_ms
                    if idle_ms >= DICTATE_IDLE_TIMEOUT_MS:
                        reason = "idle_timeout"
                        break

        # Mistaken click: timed out and never heard anything
        if reason == "idle_timeout" and not heard_speech:
            return None, reason

        # Trim trailing silence, but keep a short tail
        if reason == "idle_timeout":
            chunks = chunks[: last_speech_idx + TAIL_CHUNKS]

        # Manual stop and no speech
        if reason == "manual" and not heard_speech:
            return None, reason

        # Manual stop and speech recorded -> needs no trim

        # return single vector tensor with reason to stop
        return torch.cat(chunks), reason

    def worker(self):
        self.result["audio"], self.result["reason"] = self.record_main()

    def record(self):
        t = threading.Thread(target=self.worker)
        t.start()

        input("Dictating... press Enter to stop\n") 
        if t.is_alive():
            self.stop_event = True
        t.join()

        audio, reason = self.result["audio"], self.result["reason"]

        return audio, reason