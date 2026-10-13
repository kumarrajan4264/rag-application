# Multimodal Agentic RAG — Current RAG Core

Educational enterprise-style RAG application built incrementally from scratch.

## Current implemented pipeline

```text
PDF
 ↓
Page-level Documents
 ↓
Section Extraction
 ↓
Section-aware Chunking
 ↓
Local SentenceTransformer Embeddings
 ↓
Local ChromaDB
 ↓
Dense Retrieval + BM25
 ↓
RRF Hybrid Retrieval
 ↓
Local CrossEncoder Reranking
 ↓
Gemini
 ↓
Grounded Answer with Source/Page Citations
```

Query rewriting is also integrated before retrieval, with fallback to the original query if rewriting fails.

## Offline / VDI model requirement

The embedding and reranker models are loaded **strictly from local directories**. The code uses `local_files_only=True`, so these components will not attempt to download models from Hugging Face on the VDI.

Expected model directories:

```text
models/
├── all-MiniLM-L6-v2/
└── ms-marco-MiniLM-L-6-v2/
```

The model directories themselves are intentionally excluded from Git because they are large binary artifacts. Download them once on an internet-connected machine, copy the complete directories to the VDI, and run:

```bat
python -m scripts.check_offline_models
```

If the directories are missing, the application fails fast with a clear local-model error instead of attempting an internet download.

### Important: Gemini is currently an online dependency

The current `GeminiProvider` calls the Gemini API and therefore requires network access and an approved Gemini API key. A fully air-gapped VDI cannot run the final answer-generation stage with this provider.

For the final offline VDI deployment, the next architecture step is to replace or supplement Gemini with an approved **locally downloaded LLM**. The provider abstraction is intentionally kept separate so this can be added without changing the retrieval pipeline.

## Current validation

The development test corpus reached:
- 5 PDFs
- 451 page-level documents
- 1,584 chunks
- 1,584 Chroma vectors

A clean Chroma rebuild was used to remove stale vectors from earlier chunk-ID versions.

## Setup

```bat
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

Copy the pre-downloaded model directories into `models/` and verify:

```bat
python -m scripts.check_offline_models
```

For an environment with approved Gemini access, create `.env` from `.env.example` and add the Gemini API key.

Place PDFs in `data/raw/`.

## Example

```bat
python -m scripts.test_rag
```

## Planned next phases

1. LangGraph agentic RAG
2. Query routing and iterative retrieval
3. Multimodal PDF/image/table understanding
4. OCR/scanned-PDF routing
5. Local/offline LLM provider for the VDI
6. Retrieval and answer evaluation
7. API layer
8. Observability/tracing
9. Production hardening

## Source-control notes

- `.env`, PDFs, Chroma data, virtual environments, and downloaded model directories are intentionally excluded from source control.
- Local ChromaDB is intentional because the target VDI environment may not have internet connectivity.
- The reranker uses a local SentenceTransformers CrossEncoder model.
