import threading

from common.vad import VoiceActivityDetector
from common.transcriber import SpeechRecognizer
from common.llm import GroqLLM
from common.config import settings
from start_voice.recorder import Recorder
from start_voice.session import VoiceSession


def main():
    session = VoiceSession(
        recorder=Recorder(VoiceActivityDetector()),
        recognizer=SpeechRecognizer(),
        llm=GroqLLM(settings.groq_api_key, settings.groq_model, settings.groq_base_url),
        voice_idle_timeout_ms=settings.voice_idle_timeout_ms,
    )

    def wait_for_enter():
        input("Press Enter to stop voice chat.\n")
        session.stop()

    threading.Thread(target=wait_for_enter, daemon=True).start()
    session.run()


if __name__ == "__main__":
    main()
