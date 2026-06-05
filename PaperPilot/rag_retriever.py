import os
import logging
import requests
import chromadb
from sentence_transformers import SentenceTransformer


class RAGRetriever:
    def __init__(self, api_key: str = None, persist_dir="rag_store", top_k=5, claude_model="claude-2.1" ):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="deva_docs",
            metadata={"hnsw:space": "cosine"}
        )
        self.top_k = top_k

        # Use OpenRouter exclusively
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment. Set it in .env or the environment.")

        self.provider = "openrouter"
        self.openrouter_key = openrouter_key
        self.openrouter_base = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.openrouter_model = os.getenv("OPENROUTER_MODEL", "gpt-4o-mini")
        try:
            self.openrouter_timeout = int(os.getenv("OPENROUTER_TIMEOUT", "30"))
        except:
            self.openrouter_timeout = 30

    def add_documents(self, filename: str, chunks: list[dict]):
        if not chunks:
            return

        # Remove any existing chunks for this file so re-uploads don't clash on IDs
        self.delete_by_source(filename)

        documents = [c["text"] for c in chunks]
        metadatas = [{
            "source": filename,
            "page": c["page"],
            "line": c.get("line", 1),
            "section": c.get("section", ""),
        } for c in chunks]
        ids = [f"{filename}_{i}" for i in range(len(chunks))]

        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
        logging.info(f"Added {len(chunks)} chunks for '{filename}'")

    def query(self, question: str, history: list[dict] = None) -> str:
        try:
            chat_history = history[-10:] if history else []

            query_embed = self.embedder.encode(question).tolist()

            results = self.collection.query(
                query_embeddings=[query_embed],
                n_results=self.top_k,
                include=["documents", "metadatas"]
            )

            if not results["documents"] or not results["documents"][0]:
                return "Value not available in the source documents."

            docs = results["documents"][0]
            metas = results["metadatas"][0]

            # Build context with source labels
            context_parts = []
            for doc, meta in zip(docs, metas):
                context_parts.append(
                    f"[Source: {meta.get('source', '?')}, "
                    f"Page: {meta.get('page', '?')}, "
                    f"Line/Row: {meta.get('line', '?')}\n{doc}"
                )
            context = "\n\n---\n\n".join(context_parts)

            system_prompt = f"""You are a reliable knowledge assistant. Answer the question using ONLY the document context below.
            If the answer is not in the context, reply exactly: "Value not available in the source documents."
            Do not use outside knowledge.

            When you respond, follow this exact structure so answers are explicitly grounded:

            Answer: <DIRECT ANSWER>

            Source: <File name(s) and location(s) used — include Page and Line/Row where applicable>

            Confidence: <One-line assessment — e.g. "High: fully supported by the provided context" or "Partial: only some retrieved context supports this answer">

            Be concise. If multiple sources contributed, list them separated by semicolons.

            ## Document Context
            {context}
            """

            # Build messages for OpenRouter (OpenAI-like chat format)
            messages = [{"role": "system", "content": system_prompt}]
            for msg in chat_history:
                role = msg.get("role", "user")
                # map streamlit roles to chat roles
                if role not in ("user", "assistant"):
                    role = "user"
                messages.append({"role": role, "content": msg.get("content", "")})
            messages.append({"role": "user", "content": question})

            url = self.openrouter_base.rstrip("/") + "/chat/completions"
            headers = {"Authorization": f"Bearer {self.openrouter_key}", "Content-Type": "application/json"}
            payload = {
                "model": self.openrouter_model,
                "messages": messages,
                "max_tokens": 1500,
                "temperature": 0.4,
            }

            resp = requests.post(url, json=payload, headers=headers, timeout=self.openrouter_timeout)
            if resp.status_code != 200:
                logging.error(f"OpenRouter error: {resp.status_code} {resp.text}")
                return f"Error: OpenRouter request failed: {resp.status_code}"

            j = resp.json()
            # Try to extract content from common shapes
            content = ""
            try:
                content = j["choices"][0]["message"]["content"]
            except Exception:
                try:
                    content = j["choices"][0]["content"]
                except Exception:
                    content = j.get("choices", [{}])[0].get("message", {}).get("content", "")

            return content.strip()

        except Exception as e:
            logging.error(f"RAG query failed: {e}", exc_info=True)
            return f"Error: {e}"

    def get_doc_count(self) -> int:
        return self.collection.count()

    def list_sources(self) -> list[str]:
        all_meta = self.collection.get(include=["metadatas"])
        sources = set()
        for meta in all_meta["metadatas"]:
            sources.add(meta.get("source", "Unknown"))
        return sorted(sources)

    def delete_by_source(self, filename: str):
        all_data = self.collection.get(include=["metadatas"])
        ids_to_delete = [
            doc_id for doc_id, meta in zip(all_data["ids"], all_data["metadatas"])
            if meta.get("source") == filename
        ]
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)

    def clear_database(self):
        all_ids = self.collection.get()["ids"]
        if all_ids:
            self.collection.delete(ids=all_ids)
        logging.info("Cleared all chunks from the database")
