
import os

from google import genai

from app.config.settings import GEMINI_API_KEY, GEMINI_MODEL


class GeminiProvider:
    def __init__(self, model_name: str = GEMINI_MODEL):
        api_key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")

        if not api_key or api_key == "your_api_key_here":
            raise ValueError("A valid GEMINI_API_KEY must be configured in .env.")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")

        return response.text.strip()
