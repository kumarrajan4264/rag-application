from app.models.search import SearchResult
from app.retrieval.reranker import Reranker


def main():
    query = "What experience does the candidate have with machine learning and risk prediction?"
    results = [
        SearchResult(content="Built a Risk Prediction Machine Learning Model using Logistic Regression.", metadata={"section_title": "EXPERIENCE", "page_number": 1}),
        SearchResult(content="Management with higher number of years of experience is likely to have faced business challenges.", metadata={"section_title": "DOCUMENT", "page_number": 137}),
        SearchResult(content="Built a Machine Learning model calculating Probability of Default for risk categorization.", metadata={"section_title": "PROJECTS", "page_number": 1}),
    ]
    reranked = Reranker().rerank(query, results, top_k=3)
    for i, result in enumerate(reranked, 1):
        print(i, result.metadata, result.score)


if __name__ == "__main__":
    main()
