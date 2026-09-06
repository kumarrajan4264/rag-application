from app.rag.pipeline import RAGPipeline


rag = RAGPipeline()


question = """
What experience does the candidate have
with machine learning and risk prediction?
"""


response = rag.ask(
    question=question,
    top_k=5,
)


print("=" * 70)
print("RAG PIPELINE TEST")
print("=" * 70)

print("\nQUESTION:")
print(response["question"])

print("\nANSWER:")
print(response["answer"])

print("\n" + "=" * 70)
print("RETRIEVED SOURCES")
print("=" * 70)


for index, result in enumerate(
    response["results"],
    start=1
):

    metadata = result.metadata

    print(
        f"\n[{index}] "
        f"{metadata.get('source')} "
        f"- Page {metadata.get('page_number')}"
    )

    print(
        f"Score: {result.score}"
    )