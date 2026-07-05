# PaperPilot — Research Paper Q&A Agent

PaperPilot is a focused RAG (Retrieval-Augmented Generation) demo built on top of an existing multi-format RAG pipeline. It is tailored for answering questions about research papers (PDFs) and study notes while keeping answers explicitly grounded in source documents.

Built with: ChromaDB (vector store) + SentenceTransformer embeddings + OpenRouter `gpt-4o-mini` for generation.

## Why PaperPilot
- Demo-ready for interviews: shows ingestion, semantic chunking, embedding, similarity search, and grounded generation.
- Emphasizes anti-hallucination: the model is instructed to answer ONLY from retrieved context and to return explicit source citations and a short confidence note.

## Key Components
- `rag_ingestor.py`: multi-format file parsing and semantic chunking (PDF, DOCX, XLSX, CSV, TXT).
- `rag_retriever.py`: builds embeddings with `sentence-transformers`, stores/fetches vectors in ChromaDB, and calls the OpenAI chat API with a system prompt that enforces grounded answers with citations.
- `app.py`: Streamlit UI for uploading documents, indexing, and a chat interface to ask questions.

## Quickstart
1. Create and activate a virtual environment.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Create a `.env` file with your OpenRouter API key:
```env
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL=gpt-4o-mini
```
4. Run the app from the project root:
```bash
streamlit run PaperPilot/app.py
```

## Notes
- Keep multi-format ingestion — research PDFs are the primary demo focus.
- The system prompt in `rag_retriever.py` enforces a structured response: Answer, Source, Confidence.

---
Before pushing to a public GitHub repo:
- Remove or ignore local virtual environments (e.g. `.venv/`).
- Do NOT commit your `.env` file — it contains your OpenRouter API key. The repository contains a `.gitignore` that already excludes `.env`, `.venv/`, and `rag_store/`.
- If you want a clean remote, delete the local `rag_store/` (vector DB) and `uploaded_docs/` directories — they are intentionally excluded from the repo.

If you'd like, I can create a commit with these cleanups and prepare the push commands.

---
This project is adapted to be a personal demo (PaperPilot) for research paper Q&A, based on an earlier RAG implementation.
