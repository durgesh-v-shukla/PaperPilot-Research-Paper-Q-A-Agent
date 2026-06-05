PaperPilot — Project Notes

Overview

PaperPilot ingests documents (PDF/DOCX/XLSX/CSV/TXT), chunks them into semantic pieces, embeds the chunks with SentenceTransformer, stores vectors in ChromaDB, and answers questions by retrieving top-k similar chunks and using GPT-4o to produce grounded answers with explicit citations.

End-to-end pipeline (short)

1. Ingest
- `app.py` accepts file uploads and saves them to `uploaded_docs/`.
- `rag_ingestor.py` parses files by type and produces text "chunks" with metadata (source filename, page, line/row, section).

2. Chunking
- Text is split into reasonably sized semantic chunks (paragraph/section-aware for PDFs and DOCX; row-aware for spreadsheets).

3. Embedding & Storage
- `rag_retriever.py` uses `SentenceTransformer("all-MiniLM-L6-v2")` to embed each chunk.
- ChromaDB stores embeddings, documents, and metadata for efficient similarity search.

4. Retrieval
- On a user question, the app embeds the query, retrieves the top-k most similar chunks from ChromaDB, and concatenates them into a document context with source metadata.

5. Generation (Grounded Answer)
- The retriever builds a system prompt that instructs the model to answer ONLY from the provided context and to format the response with three parts:
  - Answer: direct response
  - Source: file/page/line citations
  - Confidence: one-line assessment of whether the context fully supports the answer
- The system prompt enforces anti-hallucination: if the answer isn't in the context the model must reply: "Value not available in the source documents.".

Notes on model configuration
- PaperPilot uses OpenRouter (`gpt-4o-mini`) as the LLM backend. Set `OPENROUTER_API_KEY` in `PaperPilot/.env` before running the app. The retriever calls OpenRouter's `/chat/completions` endpoint with a system prompt that enforces grounding and citation output.

6. UI
- Streamlit provides a sidebar for uploads and indexing, and a chat-style interface for asking questions and viewing answers with citations.

Talking points for interviews

- Emphasize the simple, explainable RAG flow: ingest → chunk → embed → store → retrieve → generate.
- Point out the anti-hallucination guardrail: the system prompt forces answers to be grounded and to cite sources.
- Note that SentenceTransformer + ChromaDB deliver real vector-based semantic search (not keyword matching).
- Explain that the confidence note is a lightweight way to signal whether the retrieved context fully supports the generated answer.

Potential small improvements (optional)

- Add more robust source-ranking (e.g., prefer longer/high-similarity chunks, aggregate multi-chunk evidence).
- Surface which specific chunk(s) produced a claim (currently the model lists file/page/line in its output if prompted).
- Add lightweight unit tests for ingestion and retrieval functions.

Repository cleanup notes
- This workspace has been cleaned for publishing: local virtualenvs and temporary DB files should be removed or ignored before pushing.
- The `.gitignore` at the project root excludes `.venv/`, `rag_store/`, and `PaperPilot/uploaded_docs/`.

That's it — minimal, explainable, and demo-friendly for research-paper Q&A.
