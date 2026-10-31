import os
import tempfile
import threading

from common.utils import save_wav

SYSTEM_PROMPT = (
    "You are Kiru, a voice assistant. Reply in short, natural spoken sentences. "
    "Do not use markdown, bullet points, or code blocks."
)
MAX_HISTORY = 20   # messages kept as context


class VoiceSession:

    def __init__(self, recorder, recognizer, llm, voice_idle_timeout_ms):
        self.recorder = recorder
        self.recognizer = recognizer
        self.llm = llm
        self.voice_idle_timeout_ms = voice_idle_timeout_ms
        self.stop_event = threading.Event()
        self.history = []

    def stop(self):
        """Manual shutdown: called by the Enter key now, by the button later."""
        self.stop_event.set()

    def run(self):
        """Blocks until shutdown. Returns 'stopped' or 'idle_timeout'."""
        self.stop_event.clear()
        self.history.clear()
        end_reason = "stopped"
        print("Voice chat started.")

        while not self.stop_event.is_set():
            audio, reason = self.recorder.listen(self.stop_event, self.voice_idle_timeout_ms)

            if audio is None:
                end_reason = reason
                break

            text = self._transcribe(audio)
            if not text:
                continue                      # noise, nothing said: listen again

            print(f"\nYou: {text}")
            self._respond(text)

        print("Voice chat ended." if end_reason == "stopped"
              else "Voice chat ended: no speech for 60 s.")
        return end_reason

    def _transcribe(self, audio):
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            save_wav(path, audio)
            return self.recognizer.transcribe(path).text.strip()
        finally:
            os.remove(path)

    def _respond(self, text):
        self.history.append({"role": "user", "content": text})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history[-MAX_HISTORY:]

        reply = []
        print("Kiru: ", end="")
        for token in self.llm.stream_chat(messages):
            if self.stop_event.is_set():
                break
            reply.append(token)
            print(token, end="", flush=True)
            # TODO next step: cut tokens into sentences and send them to TTS
        print()

        if reply:
            self.history.append({"role": "assistant", "content": "".join(reply)})