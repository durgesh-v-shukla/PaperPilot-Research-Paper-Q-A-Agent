import os
import time
import streamlit as st
from dotenv import load_dotenv
from rag_ingestor import FileIngestor
from rag_retriever import RAGRetriever

load_dotenv()

st.set_page_config(page_title="PaperPilot — Research Paper Q&A Agent", layout="wide")

# --- Styling: refined professional look ---
st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
        :root { --bg:#0b0f13; --card:#0f1720; --muted:#94a3b8; --accent:#0b3d91; --accent-dark:#062f6d; --soft:#08121a; --text:#ffffff; }
        html, body { font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; background: var(--bg); color: var(--text); }
        .pp-header { display:flex; align-items:center; gap:16px; padding:6px 0; }
        .pp-brand { font-size:28px; font-weight:700; color:var(--accent) !important; letter-spacing:0.2px }
        .pp-sub { color:var(--text) !important; font-size:14px; margin-top:2px; }
        .pp-card { background: var(--card); border-radius:10px; padding:14px; box-shadow:0 6px 18px rgba(2,6,23,0.6); color:var(--text); }
        .pp-doc { padding:10px 12px; border-radius:8px; background:rgba(255,255,255,0.03); margin-bottom:8px; color:var(--text); font-weight:500 }
        /* Make file uploader blend with sidebar (remove large white box) */
        [data-testid="stFileUploader"] > div {
            background: transparent !important;
            box-shadow: none !important;
            padding: 0 !important;
        }
        [data-testid="stFileUploader"] .stFileUpload {
            background: transparent !important;
        }
        /* Additional fallbacks to hide the uploader white background */
        [data-testid="stFileUploader"] { background: transparent !important; padding: 0 !important; }
        [data-testid="stFileUploader"] div[role="button"] { background: transparent !important; box-shadow:none !important; }
        .stSidebar { padding-top: 8px; }
        .stButton>button { background: linear-gradient(90deg,var(--accent), var(--accent-dark)); color: white; border-radius:8px; padding:8px 12px; border: none; }
        /* Chat bubbles */
        .chat-user { background: linear-gradient(90deg,#0b3d91,#062f6d); padding:10px 12px; border-radius:12px; color:#ffffff }
        .chat-assistant { background:#08121a; padding:10px 12px; border-radius:12px; box-shadow:0 2px 6px rgba(2,6,23,0.04); color:#ffffff }
        /* Make markdown content slightly larger for readability (default color) */
        .stMarkdown { font-size:15px; color:var(--text) }
        /* Ensure standard Streamlit headers, captions, and text use app text color */
        .stHeader, .stSubheader, .stCaption, .stText, .css-1v3fvcr, .css-10trblm { color: var(--text) !important }
        /* Keep UI card and header text dark for contrast */
        .pp-card, .pp-brand, .pp-sub, .pp-doc, .stSidebar, .stCaption { color: var(--text) !important }
        /* Ensure chat message text is white for both user and assistant bubbles */
        div[data-testid^="stChat"] .stMarkdown, div[data-testid="stChatMessage"] .stMarkdown, .stChatMessage .stMarkdown, div[class*="stChat"] .stMarkdown { color: #ffffff !important }
        div[data-testid^="stChat"] .css-1v3fvcr, div[data-testid="stChatMessage"] .css-1v3fvcr, .stChatMessage .css-1v3fvcr { color: #ffffff !important }
        /* Ensure uploader heading is visible */
        .pp-card strong { color: var(--text); font-size:15px }
        /* Ensure Streamlit container text is readable */
        .css-1d391kg, .css-10trblm, .css-1v3fvcr { color: var(--text) !important }
        hr.pp-sep { border:none; border-top:1px solid rgba(3,105,102,0.06); margin:14px 0 18px 0 }
        </style>
        """,
        unsafe_allow_html=True,
)

st.markdown(
        """
        <div class='pp-header'>
            <div class='pp-brand' style='color:var(--accent)'>PaperPilot</div>
            <div style='flex:1'></div>
            <div style='text-align:right'>
                <div class='pp-sub'>Research Paper Q&A — Grounded, citation-first answers</div>
            </div>
        </div>
        <hr class='pp-sep' />
        """,
        unsafe_allow_html=True,
)

os.makedirs("uploaded_docs", exist_ok=True)


# --- API Key management UI ---
def write_env(updates: dict):
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    # Load existing
    existing = {}
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.strip().split("=", 1)
                    existing[k] = v

    # Update and write back
    existing.update(updates)
    with open(env_path, "w", encoding="utf-8") as f:
        for k, v in existing.items():
            f.write(f"{k}={v}\n")
    # Reload environment variables into the running process
    try:
        load_dotenv(env_path, override=True)
    except Exception:
        pass


# Sidebar messaging removed per user preference


# --- Init ---
@st.cache_resource
def get_retriever(api_key: str):
    if not api_key:
        return None
    return RAGRetriever(api_key=api_key)


rag = get_retriever(os.getenv("OPENROUTER_API_KEY"))
ingestor = FileIngestor()

# --- Sidebar: Upload & Manage Documents ---
with st.sidebar:
    st.header("Documents")

    uploaded_files = st.file_uploader(
        "Upload documents — Supported: PDF, DOCX, XLSX, CSV, TXT",
        type=["pdf", "docx", "xlsx", "csv", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join("uploaded_docs", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            chunks, filename = ingestor.process_file(file_path)

            if chunks:
                if rag is None:
                    st.warning("No LLM configured yet — set OPENROUTER_API_KEY in .env and restart the app to enable querying.")
                else:
                    rag.add_documents(filename=uploaded_file.name, chunks=chunks)
                    st.success(f"{uploaded_file.name} - {len(chunks)} chunks indexed")
            else:
                st.warning(f"No content extracted from {uploaded_file.name}")

    st.divider()
    st.subheader("Indexed Documents")
    if rag is None:
        st.info("No retriever configured. Set OPENROUTER_API_KEY in .env and refresh the app to enable querying.")
    else:
        sources = rag.list_sources()
        if sources:
            for src in sources:
                st.markdown(f"<div class='pp-doc'>{src}</div>", unsafe_allow_html=True)
            st.caption(f"Total chunks: {rag.get_doc_count()}")
        else:
            st.info("No documents uploaded yet.")

    st.divider()
    if st.button("Clear All Data"):
        if st.button("Confirm: delete all indexed data and uploaded files"):
            if rag is not None:
                rag.clear_database()
            for f_name in os.listdir("uploaded_docs"):
                os.remove(os.path.join("uploaded_docs", f_name))
            st.success("Database cleared!")
            st.experimental_set_query_params(_refresh=str(time.time()))


# --- Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask a question about your research papers..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if rag is None:
        answer = "No LLM configured. Set OPENROUTER_API_KEY in .env and refresh the app."
    else:
        if rag.get_doc_count() == 0:
            answer = "Please upload some documents first using the sidebar."
        else:
            with st.spinner("Searching across your documents..."):
                answer = rag.query(question, history=st.session_state.messages[:-1])

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
