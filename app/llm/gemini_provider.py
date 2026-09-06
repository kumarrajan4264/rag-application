from google import genai

from app.config.settings import GEMINI_API_KEY


class GeminiProvider:

    MODEL_NAME = "gemini-3.5-flash-lite"

    def __init__(self):

        self.client = genai.Client(
            api_key=GEMINI_API_KEY
        )

    def generate(
        self,
        prompt: str,
    ) -> str:

        interaction = self.client.interactions.create(
            model=self.MODEL_NAME,
            input=prompt,
        )

        return interaction.output_text