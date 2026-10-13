from app.llm.gemini_provider import GeminiProvider


class QueryRewriter:
    def __init__(self):
        self.llm = GeminiProvider()

    def rewrite(self, query: str) -> str:
        prompt = f"""
You are a query rewriting component for an enterprise document retrieval system.

Rewrite the user's question into a single clear search query that maximizes retrieval of relevant information.

Rules:
- Preserve the original meaning.
- Do not answer the question.
- Do not add facts that are not present in the question.
- Expand vague references when possible.
- Include important concepts and entities from the query.
- Return ONLY the rewritten search query.
- Do not use bullets, explanations, or quotation marks.

USER QUERY:
{query}

REWRITTEN QUERY:
"""
        try:
            rewritten_query = self.llm.generate(prompt).strip()
            return rewritten_query or query
        except Exception:
            return query
