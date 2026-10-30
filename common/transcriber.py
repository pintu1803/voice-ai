from faster_whisper import WhisperModel
from start_voice.transcription_result import TranscriptionResult
from config import (MODEL_PATH, COMPUTE_TYPE)

class SpeechRecognizer:
    def __init__(self):
        self.model = WhisperModel(
                    MODEL_PATH,
                    device="cpu",
                    compute_type=COMPUTE_TYPE
                )

    def transcribe(self, filename: str, language : str | None = "en"):
        print("Transcribing...")
        segments, info = self.model.transcribe(filename,
                                               language=language,
                                               vad_filter=False, #not enabling in-built VAD of whisper
                                               condition_on_previous_text=False,
                                               beam_size=5)

        print(f"Language: {info.language} ({info.language_probability:.2f})")

        result_segments = []
        full_text = ""

        for segment in segments:
            result_segments.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text
                }
            )
            full_text += segment.text + " "

        return TranscriptionResult(
            text=full_text.strip(),
            language=info.language,
            segments=result_segments
        )