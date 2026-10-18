import torch
from scipy.io.wavfile import read, write

from silero_vad import (
    load_silero_vad,
    get_speech_timestamps,
    collect_chunks
)


class VoiceActivityDetector:

    def __init__(self):

        self.model = load_silero_vad()


    def remove_silence(
        self,
        input_file: str,
        output_file: str
    ):

        # Load audio
        sample_rate, audio = read(input_file)
        audio = audio.squeeze()
        wav = torch.tensor(audio, dtype=torch.float32)
        wav = wav / 32768.0

        print("Sample rate:", sample_rate)
        print("Audio shape:", wav.shape)

        # Detect speech regions
        speech_timestamps = get_speech_timestamps(
            wav,
            self.model,
            sampling_rate=sample_rate
        )

        print("Speech regions:")
        print(speech_timestamps)

        speech_audio = collect_chunks(speech_timestamps, wav)
        speech_audio = (speech_audio.numpy() * 32767).astype("int16")

        write(output_file, sample_rate, speech_audio)