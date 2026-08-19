import os
import time
import shutil

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

DOCS_DIR = "data/docs"
os.makedirs(DOCS_DIR, exist_ok=True)

SYSTEM_PROMPT = """
    You are an assistant answering questions using the document.

    If the user asks multiple questions, treat each question independently.

    For EACH question:
    1. Call the retrieval_function separately using that question.
    2. Use the retrieved context to answer that question.
    3. Do not reuse the retrieved context from another question.

    Answer only using information returned by the retrieval_function.
    If the required information cannot be found, say "I don't know".
"""


def document_process(path: str, checkpointer=None, progress_callback=None):
    """
    Loads every PDF in `path`, splits it, embeds it, builds an in-memory
    vector store + retrieval tool, and returns a ready-to-use agent.

    This is the user's original document_process() logic, with two bug
    fixes: `tools=[retriever]` -> `tools=[retrieval_function]` (retriever
    was undefined), and the built agent is now actually returned instead
    of being assigned to st.session_state outside the function's scope.
    """

    def _report(msg):
        if progress_callback:
            progress_callback(msg)

    # Document loading
    _report("Loading PDFs from data/docs ...")
    loader = PyPDFDirectoryLoader(path)
    pdf = loader.load()

    if not pdf:
        raise ValueError("No readable PDF content found in the docs folder.")

    # Text Splitting
    _report("Splitting text into chunks ...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    split_text = splitter.split_documents(pdf)

    # Vector embeddings
    _report("Embedding chunks with all-MiniLM-L6-v2 ...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector db
    _report("Building the in-memory vector store ...")
    vector_store = InMemoryVectorStore.from_documents(
        documents=split_text,
        embedding=embedding_model,
    )

    @tool
    def retrieval_function(question: str):
        """
        This tool helps in retrieving relevant information from the Document.
        This tool returns the relevant info (context).
        """
        docs = vector_store.similarity_search(query=question, k=3)
        context = ""
        for i in docs:
            context += i.page_content + "\n"
        return context

    _report("Spinning up the agent ...")
    llm = ChatGroq(model="openai/gpt-oss-20b")

    agent = create_agent(
        model=llm,
        tools=[retrieval_function],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer or InMemorySaver(),
    )

    return agent, len(pdf), len(split_text)


# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="DocChat — PDF RAG Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Styling
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: radial-gradient(circle at top left, #10131c 0%, #0b0d13 60%); }

    /* Hero header */
    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(124,58,237,0.18), rgba(37,99,235,0.12));
        border: 1px solid rgba(148,163,184,0.15);
        margin-bottom: 1.4rem;
    }
    .hero h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p { color: #94a3b8; margin-top: 0.3rem; font-size: 0.95rem; }

    /* Status pill */
    .pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .pill-ready { background: rgba(34,197,94,0.15); color: #4ade80; border: 1px solid rgba(74,222,128,0.3); }
    .pill-idle { background: rgba(148,163,184,0.15); color: #94a3b8; border: 1px solid rgba(148,163,184,0.3); }

    /* Uploaded file chip */
    .file-chip {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0.75rem;
        border-radius: 10px;
        background: rgba(148,163,184,0.08);
        border: 1px solid rgba(148,163,184,0.15);
        margin-bottom: 0.4rem;
        font-size: 0.85rem;
        color: #e2e8f0;
        animation: slideIn 0.35s ease;
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-8px); }
        to { opacity: 1; transform: translateX(0); }
    }

    /* Chat bubbles get a subtle fade-in */
    [data-testid="stChatMessage"] { animation: fadeIn 0.4s ease; }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0d0f17;
        border-right: 1px solid rgba(148,163,184,0.1);
    }

    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        border: none;
        background: linear-gradient(90deg, #7c3aed, #2563eb);
        color: white;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(124,58,237,0.35);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------
defaults = {
    "document_uploaded": False,
    "agent": None,
    "chat_history": [],
    "memory": InMemorySaver(),
    "thread_id": "session-1",
    "doc_stats": None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ------------------------------------------------------------------
# Sidebar — upload & process
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📁 Document Manager")

    status = (
        '<span class="pill pill-ready">● Agent ready</span>'
        if st.session_state.document_uploaded
        else '<span class="pill pill-idle">● Not processed</span>'
    )
    st.markdown(status, unsafe_allow_html=True)
    st.write("")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type="pdf",
        accept_multiple_files=True,
        help="Files are saved to data/docs/",
    )

    if uploaded_files:
        with st.spinner("Saving files ..."):
            for f in uploaded_files:
                dest = os.path.join(DOCS_DIR, f.name)
                with open(dest, "wb") as out:
                    out.write(f.getbuffer())
                time.sleep(0.05)
        st.success(f"Saved {len(uploaded_files)} file(s) to `{DOCS_DIR}`")

    existing_docs = sorted(os.listdir(DOCS_DIR)) if os.path.exists(DOCS_DIR) else []

    if existing_docs:
        st.markdown("**Documents in `data/docs/`**")
        for doc in existing_docs:
            st.markdown(
                f'<div class="file-chip">📄 {doc}</div>', unsafe_allow_html=True
            )
    else:
        st.caption("No documents uploaded yet.")

    st.write("")
    col_a, col_b = st.columns(2)

    with col_a:
        process_clicked = st.button(
            "⚙️ Process",
            use_container_width=True,
            disabled=not existing_docs,
        )
    with col_b:
        if st.button("🗑️ Clear all", use_container_width=True, disabled=not existing_docs):
            shutil.rmtree(DOCS_DIR, ignore_errors=True)
            os.makedirs(DOCS_DIR, exist_ok=True)
            st.session_state.document_uploaded = False
            st.session_state.agent = None
            st.session_state.chat_history = []
            st.session_state.doc_stats = None
            st.rerun()

    if process_clicked:
        progress_box = st.empty()
        bar = st.progress(0, text="Starting ...")

        steps = [
            "Loading PDFs from data/docs ...",
            "Splitting text into chunks ...",
            "Embedding chunks with all-MiniLM-L6-v2 ...",
            "Building the in-memory vector store ...",
            "Spinning up the agent ...",
        ]

        def on_progress(msg):
            try:
                idx = steps.index(msg)
            except ValueError:
                idx = 0
            bar.progress(int((idx + 1) / len(steps) * 90), text=msg)

        try:
            with progress_box.container():
                st.markdown("🧠 **Processing documents ...**")
                agent, n_pages, n_chunks = document_process(
                    DOCS_DIR,
                    checkpointer=st.session_state.memory,
                    progress_callback=on_progress,
                )
            bar.progress(100, text="Done!")
            time.sleep(0.3)
            bar.empty()
            progress_box.empty()

            st.session_state.agent = agent
            st.session_state.document_uploaded = True
            st.session_state.doc_stats = {"pages": n_pages, "chunks": n_chunks}
            st.session_state.chat_history = []
            st.toast("Documents indexed — ask away!", icon="✅")
        except Exception as e:
            bar.empty()
            progress_box.empty()
            st.error(f"Something went wrong while processing: {e}")

    if st.session_state.doc_stats:
        st.write("")
        st.markdown("**Index stats**")
        s1, s2 = st.columns(2)
        s1.metric("Pages", st.session_state.doc_stats["pages"])
        s2.metric("Chunks", st.session_state.doc_stats["chunks"])

# ------------------------------------------------------------------
# Main — hero + chat
# ------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>📄 DocChat</h1>
        <p>Upload PDFs on the left, hit <b>Process</b>, then ask anything about them.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.document_uploaded:
    st.info("👈 Upload one or more PDFs in the sidebar and click **Process** to get started.")
else:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask a question about your documents ...")

    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.spinner("Thinking ..."):
                response = st.session_state.agent.invoke(
                    {"messages": [{"role": "user", "content": prompt}]},
                    config={"configurable": {"thread_id": st.session_state.thread_id}},
                )
                answer = response["messages"][-1].content
            placeholder.markdown(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})