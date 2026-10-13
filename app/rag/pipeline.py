from app.config.settings import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIRECTORY
from app.embeddings.embedding_model import EmbeddingModel
from app.llm.gemini_provider import GeminiProvider
from app.rag.query_rewriter import QueryRewriter
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker
from app.vector_store.chroma_store import ChromaVectorStore


class RAGPipeline:
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_store = ChromaVectorStore(CHROMA_PERSIST_DIRECTORY, CHROMA_COLLECTION_NAME)
        chunks = self.vector_store.get_all_chunks()
        self.bm25_retriever = BM25Retriever(chunks=chunks)
        self.hybrid_retriever = HybridRetriever(self.vector_store, self.bm25_retriever)
        self.reranker = Reranker()
        self.query_rewriter = QueryRewriter()
        self.llm = GeminiProvider()

    def retrieve(self, question: str, top_k: int = 5, metadata_filter: dict | None = None):
        rewritten_query = self.query_rewriter.rewrite(question)
        query_vector = self.embedding_model.embed_text(rewritten_query)
        candidates = self.hybrid_retriever.search(
            query_vector=query_vector,
            query=rewritten_query,
            limit=max(top_k * 2, 10),
            metadata_filter=metadata_filter,
        )
        return self.reranker.rerank(rewritten_query, candidates, top_k)

    def build_context(self, results):
        parts = []
        for index, result in enumerate(results, start=1):
            metadata = result.metadata
            parts.append(
                f"SOURCE {index}\n"
                f"Source: {metadata.get('source', 'Unknown')}\n"
                f"Page: {metadata.get('page_number', 'Unknown')}\n\n"
                f"Content:\n{result.content}\n"
            )
        return "\n".join(parts)

    def generate_answer(self, question: str, context: str):
        prompt = f"""
You are an enterprise document analysis assistant.

Answer the user's question ONLY using the provided context.
Do not use outside knowledge.
If the answer cannot be found in the context, say:
"I don't have enough information in the provided documents."
Always cite the source and page number when answering.

USER QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

ANSWER:
"""
        return self.llm.generate(prompt)

    def ask(self, question: str, top_k: int = 5):
        results = self.retrieve(question, top_k)
        context = self.build_context(results)
        answer = self.generate_answer(question, context)
        return {"question": question, "answer": answer, "results": results}
