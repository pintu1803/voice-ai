import os
import tempfile
import threading
import pyttsx3
import re, queue
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
        self.engine = pyttsx3.init()
        self.setSpeakerAsZira()

    def setSpeakerAsZira(self):
        voices = self.engine.getProperty('voices')
        for v in voices:
            print(v.id, v.name)
            if "zira" in v.name.lower():
                self.engine.setProperty("voice", v.id)
                print("Set voice to : ", v.id)
                break

        print("Speech rate:", self.engine.getProperty("rate"))
        self.engine.setProperty("rate", 170)
        print("Set speech rate : ", self.engine.getProperty("rate"))

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

    def _respond1(self, text):
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
            speakText = "".join(reply).strip()
            self.engine.say(speakText)
            self.engine.runAndWait()
            self.history.append({"role": "assistant", "content": "".join(reply)})

    def _respond(self, text):
        self.history.append({"role": "user", "content": text})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ] + self.history[-MAX_HISTORY:]

        speech_queue = queue.Queue()
        cancelled = threading.Event()
        reply_parts = []

        def generate():
            pending = ""

            try:
                for token in self.llm.stream_chat(messages):
                    if cancelled.is_set() or self.stop_event.is_set():
                        return

                    reply_parts.append(token)
                    pending += token

                    # A sentence ends at punctuation followed by whitespace.
                    boundary = re.search(r"[.!?](?=\s)", pending)

                    while boundary:
                        end = boundary.end()
                        sentence = pending[:end].strip()

                        speech_queue.put(("sentence", sentence))
                        pending = pending[end:].lstrip()

                        boundary = re.search(r"[.!?](?=\s)", pending)

                # Speak the final text even if it has no punctuation.
                if pending.strip():
                    speech_queue.put(("sentence", pending.strip()))

            except Exception as exc:
                speech_queue.put(("error", exc))

            finally:
                speech_queue.put(("done", None))

        current_sentence = ""
        printed_to = 0

        def on_word(name, location, length):
            nonlocal printed_to

            if self.stop_event.is_set():
                self.engine.stop()
                return

            end = location + length

            # Reject invalid positions instead of clamping them.
            if length <= 0 or location < 0 or end > len(current_sentence):
                return

            # Real word events contain the word here.
            # The incorrectly reported stream-start event does not.
            expected_word = current_sentence[location:end]
            if name != expected_word:
                return

            if end > printed_to:
                print(
                    current_sentence[printed_to:end],
                    end="",
                    flush=True,
                )
                printed_to = end

        connection = self.engine.connect("started-word", on_word)
        print("Kiru: ", end="", flush=True)

        try:
            threading.Thread(target=generate, daemon=True).start()

            while not self.stop_event.is_set():
                try:
                    kind, payload = speech_queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                if kind == "done":
                    break

                if kind == "error":
                    raise payload

                current_sentence = payload
                printed_to = 0

                self.engine.say(current_sentence)
                self.engine.runAndWait()

                if not self.stop_event.is_set():
                    remaining = current_sentence[printed_to:]

                    # Only flush trailing punctuation and whitespace.
                    if not any(character.isalnum() for character in remaining):
                        print(remaining, end="", flush=True)

                    print(" ", end="", flush=True)

        finally:
            cancelled.set()
            self.engine.disconnect(connection)
            print()

        if reply_parts and not self.stop_event.is_set():
            self.history.append({
                "role": "assistant",
                "content": "".join(reply_parts),
            })