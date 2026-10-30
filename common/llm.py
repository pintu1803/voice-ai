from openai import OpenAI

class GroqLLM():

    def __init__(self, api_key:str, model:str, base_url:str):
        self.model = model
        self.client = OpenAI(api_key=api_key,
                             base_url=base_url)

    def generate(self, prompt:str)-> str:

        try:           
            response = self.client.chat.completions.create(
                            model=self.model,
                            messages=[
                                        {
                                            "role": "system",
                                            "content": """
                                            You are a Java backend interview assistant.
                                            Use the retrieved context.
                                            Explain concepts with examples.
                                            """
                                        },
                                        {
                                            "role": "user",
                                            "content": prompt
                                        }
                                    ],
                            temperature=0.2
                        )
            content = response.choices[0].message.content if response.choices else None 
            if not content:
                raise RuntimeError("Groq returned no text")
            return str(content)
        
        except Exception as e:
            print("GroqLLM generate failed: %s", e)
            return "Model is unavailable"
