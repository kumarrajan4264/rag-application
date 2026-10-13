from app.rag.query_rewriter import QueryRewriter


def main():
    rewriter = QueryRewriter()
    queries = [
        "Tell me about his ML work.",
        "What did he do at FlexiLoans?",
        "What experience does the candidate have with machine learning and risk prediction?",
    ]
    for query in queries:
        print("ORIGINAL:", query)
        print("REWRITTEN:", rewriter.rewrite(query))
        print("-" * 70)


if __name__ == "__main__":
    main()
