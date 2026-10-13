from app.rag.pipeline import RAGPipeline


def main():
    pipeline = RAGPipeline()
    question = "What experience does the candidate have with machine learning and risk prediction?"
    result = pipeline.ask(question)
    print("=" * 70)
    print("RAG PIPELINE TEST")
    print("=" * 70)
    print("\nQUESTION:\n")
    print(question)
    print("\nANSWER:\n")
    print(result["answer"])
    print("\nRETRIEVED SOURCES\n")
    for index, item in enumerate(result["results"], 1):
        print(f"[{index}] {item.metadata.get('source')} - Page {item.metadata.get('page_number')}")
        print(f"Score: {item.score}")


if __name__ == "__main__":
    main()
