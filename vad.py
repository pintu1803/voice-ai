import torch
from dataclasses import dataclass
from silero_vad import (load_silero_vad, VADIterator)

from config import (SPEECH_PROB_THRESHOLD, SAMPLE_RATE)

@dataclass
class VADEvent:
    speech_started: bool = False
    speech_ended: bool = False
    is_speech: bool = False


class VoiceActivityDetector:

    def __init__(self): 
        self.model = load_silero_vad()

        self.iterator = VADIterator(
            self.model,
            threshold=SPEECH_PROB_THRESHOLD,
            sampling_rate=SAMPLE_RATE
        )


    def process(self, chunk: torch.Tensor) -> VADEvent:

        result = self.iterator(chunk)
        event = VADEvent(is_speech=False)

        if result is None:
            return event

        if "start" in result:
            event.speech_started = True

        if "end" in result:
            event.speech_ended = True

        return event