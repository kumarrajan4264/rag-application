# Manager Handoff — RAG Application

## Current scope

This repository contains the current working RAG core built as an educational enterprise-style implementation.

Current pipeline:

```text
PDF → Page Documents → Sections → Chunks → Local Embeddings → ChromaDB
    → Dense + BM25 → RRF Hybrid → CrossEncoder Reranker → Gemini
```

Query rewriting is performed before retrieval, with a fallback to the original query if rewriting fails.

## Validated development result

The test corpus contained 5 PDFs, 451 page-level documents, 1,584 chunks, and exactly 1,584 vectors after a clean Chroma rebuild.

The retrieval pipeline was tested through dense retrieval, BM25, hybrid/RRF retrieval, reranking, and end-to-end RAG answer generation.

## VDI / offline design

The embedding and reranker models are designed to run locally from:

```text
models/all-MiniLM-L6-v2
models/ms-marco-MiniLM-L-6-v2
```

Both loaders use `local_files_only=True`, so they do not depend on Hugging Face internet access at runtime.

The model binaries are intentionally not committed to Git. They should be downloaded once on an approved internet-connected machine and transferred to the VDI through the organization's approved process.

### Current limitation

Gemini is still an online API dependency. Therefore, the current end-to-end answer generation is suitable for an environment with approved Gemini connectivity, but not for a fully air-gapped VDI.

The planned next step is a local LLM provider using a model pre-downloaded and approved for the VDI. This can be added behind the existing LLM provider boundary without redesigning ingestion, retrieval, hybrid search, or reranking.

## Planned architecture

```text
Document
  ↓
Page / Semantic Blocks
  ↓
Sections / Chunks
  ↓
Local Embeddings
  ↓
Local Vector Store
  ↓
Query Rewrite / Routing
  ↓
Hybrid Retrieval
  ↓
Reranking
  ↓
LangGraph Agentic Retrieval
  ↓
Local LLM / approved LLM provider
  ↓
Grounded Answer + Citations
```

Future phases include multimodal document understanding, OCR/scanned-PDF handling, evaluation, API exposure, observability, and production hardening.
