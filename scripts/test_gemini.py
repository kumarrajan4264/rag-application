from app.llm.gemini_provider import GeminiProvider


llm = GeminiProvider()


response = llm.generate(
    """
    Explain Retrieval Augmented Generation
    in three simple sentences.
    """
)


print("=" * 70)
print("GEMINI TEST")
print("=" * 70)

print(response)