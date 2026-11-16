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

        #text: full text of the audio file gets converted into one string
        #language: english or whatever
        #segments is an array of dictionary, each dict contains text, start time and end time