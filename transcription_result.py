class TranscriptionResult:

    def __init__(
        self,
        text,
        language,
        segments
    ):
        self.text = text
        self.language = language
        self.segments = segments