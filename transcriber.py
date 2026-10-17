from faster_whisper import WhisperModel


class SpeechRecognizer:
    def __init__(self):
        self.model = WhisperModel(
                    "./models/faster-whisper-base",
                    device="cpu",
                    compute_type="int8"
                )

    def transcribe(self, filename: str):
        print("Transcribing...")
        segments, info = self.model.transcribe(filename)

        print()

        print("Detected language:", info.language)

        print()

        text = " ".join(segment.text for segment in segments)
        
        return text