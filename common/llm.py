from openai import OpenAI
from typing import Iterator

class GroqLLM():

    def __init__(self, api_key:str, model:str, base_url:str):
        self.model = model
        self.client = OpenAI(api_key=api_key,
                             base_url=base_url)

    def stream(self, prompt:str)-> Iterator[str]:

        try:           
            response = self.client.chat.completions.create(
                            model=self.model,
                            messages=[
                                        {
                                            "role": "system",
                                            "content": """
                                            "You are Kiru, a voice assistant. Reply in short, natural spoken sentences. "
                                            "Do not use markdown, bullet points, or code blocks."
                                            """
                                        },
                                        {"role": "user", "content": prompt}                                     
                                    ],
                            temperature=0.2,
                            stream=True
                        )
            for chunk in response:
                token = chunk.choices[0].delta.content
                if token:
                    yield token

        except Exception as e:
            print("GroqLLM generate failed: %s", e)
            yield "Model is unavailable"


    def stream_chat(self, messages: list[dict]) -> Iterator[str]:
        try:
            response = self.client.chat.completions.create( # pyright: ignore[reportCallIssue, reportArgumentType]
                        model=self.model,
                        messages=messages, # pyright: ignore[reportCallIssue, reportArgumentType]
                        temperature=0.4,
                        stream=True
            )
            for chunk in response:
                if not chunk.choices:
                    continue
                token = chunk.choices[0].delta.content
                if token:
                    yield token
        except Exception:
            print("GroqLLM stream_chat failed")
            yield "Sorry, I couldn't reach the model."