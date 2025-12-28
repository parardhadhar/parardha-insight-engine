import streamlit as st
import os
import tempfile
import shutil
from huggingface_hub import snapshot_download

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Parardha's Insight Engine",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.title-text {
    text-align: center;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(45deg, #FF4B4B, #0068C9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
    padding-top: 20px;
}

.subtitle-text {
    text-align: center;
    font-size: 1.1rem;
    font-weight: 400;
    opacity: 0.8;
    margin-bottom: 40px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2593/2593491.png", width=60)
    st.markdown("### **Insight Engine**")
    st.caption("v1.0 • Built by **Parardha Dhar**")
    st.markdown("---")

    st.markdown("#### 1️⃣ System Status")
    api_key = os.getenv("GROQ_API_KEY")

    if api_key:
        st.success("✅ System Online")
    else:
        st.error("⚠️ GROQ_API_KEY missing")
        st.stop()

    st.markdown("---")
    st.markdown("#### 2️⃣ Upload PDF")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        st.caption(f"📄 {uploaded_file.name}")

# ---------------- HEADER ----------------
st.markdown('<div class="title-text">Parardha\'s Insight Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Upload a document and talk to your data.</div>', unsafe_allow_html=True)

# ---------------- CHAT STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload a PDF to begin."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- MODEL LOADER ----------------
@st.cache_resource(show_spinner=False)
def load_models(_api_key):
    try:
        llm = Groq(model="llama-3.3-70b-versatile", api_key=_api_key)

        repo_id = "sentence-transformers/all-MiniLM-L6-v2"
        model_path = "/tmp/ai_model"

        if not os.path.exists(model_path):
            snapshot_download(repo_id=repo_id, local_dir=model_path, local_dir_use_symlinks=False)

        embed_model = HuggingFaceEmbedding(model_name=model_path)

        return llm, embed_model, None
    except Exception as e:
        return None, None, str(e)

# ---------------- INDEXING ----------------
def index_document(file):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, file.name)
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())

        reader = SimpleDirectoryReader(input_dir=temp_dir)
        documents = reader.load_data()
        return VectorStoreIndex.from_documents(documents)

# ---------------- CHAT INPUT ----------------
if prompt := st.chat_input("Ask something about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not uploaded_file:
        st.warning("Upload a PDF first.")
        st.stop()

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            llm, embed_model, error = load_models(api_key)

            if error:
                st.error(error)
                st.stop()

            Settings.llm = llm
            Settings.embed_model = embed_model

            if "vector_index" not in st.session_state:
                st.session_state.vector_index = index_document(uploaded_file)

            engine = st.session_state.vector_index.as_query_engine()
            response = engine.query(prompt)

            st.markdown(response.response)
            st.session_state.messages.append({"role": "assistant", "content": response.response})

# ---------------- RAILWAY PORT BIND ----------------
import subprocess

if __name__ == "__main__":
    port = os.getenv("PORT", "8501")
    subprocess.run([
        "streamlit", "run", "app.py",
        "--server.port", port,
        "--server.address", "0.0.0.0"
    ])
