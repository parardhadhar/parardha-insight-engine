import streamlit as st
import os
import tempfile
import shutil
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. Load Environment Variables
load_dotenv()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Parardha's Insight Engine",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR AESTHETICS ---
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Custom Header Styling */
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

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2593/2593491.png", width=60)
    st.markdown("### **Insight Engine**")
    st.caption("v1.0 • Built by **Parardha**")
    st.markdown("---")
    
    # Step 1: API Key
    st.markdown("#### 1️⃣ Configuration")
    api_key = os.getenv("GROQ_API_KEY")
    
    if api_key:
        st.success("✅ System Online")
    else:
        st.error("⚠️ API Key Not Found")
        st.info("Please set `GROQ_API_KEY` in secrets.")
        st.stop()
    
    st.markdown("---")
    
    # Step 2: Upload
    st.markdown("#### 2️⃣ Knowledge Base")
    uploaded_file = st.file_uploader("Upload a PDF to analyze", type=["pdf"])
    
    if uploaded_file:
        st.caption(f"📄 Loaded: {uploaded_file.name}")
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; opacity: 0.6; font-size: 0.8em;">
        Powered by <b>Llama 3.3</b> & <b>Parardha's AI</b>
    </div>
    """, unsafe_allow_html=True)

# --- MAIN HEADER ---
st.markdown('<div class="title-text">Parardha\'s Insight Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Your personal AI research assistant. Upload a document to unlock insights.</div>', unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Greetings! I am ready to analyze your documents. Please upload a PDF in the sidebar to begin."}
    ]

# --- DISPLAY CHAT HISTORY ---
for message in st.session_state.messages:
    # Custom avatars for a polished look
    avatar = "https://cdn-icons-png.flaticon.com/512/4712/4712027.png" if message["role"] == "assistant" else "https://cdn-icons-png.flaticon.com/512/9131/9131529.png"
    
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# --- MODEL LOADING LOGIC ---
@st.cache_resource(show_spinner=False)
def load_models(_api_key):
    try:
        # LLM Setup
        llm = Groq(model="llama-3.3-70b-versatile", api_key=_api_key)
        
        # Embedding Setup (Stable)
        repo_id = "sentence-transformers/all-MiniLM-L6-v2"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "ai_model")
        
        if not os.path.exists(model_path):
             snapshot_download(repo_id=repo_id, local_dir=model_path, local_dir_use_symlinks=False)
        
        try:
            embed_model = HuggingFaceEmbedding(model_name=model_path)
        except Exception:
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
            snapshot_download(repo_id=repo_id, local_dir=model_path, local_dir_use_symlinks=False)
            embed_model = HuggingFaceEmbedding(model_name=model_path)
        
        return llm, embed_model, None
    except Exception as e:
        return None, None, str(e)

def index_document(file):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, file.name)
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())
            
        reader = SimpleDirectoryReader(input_dir=temp_dir)
        documents = reader.load_data()
        index = VectorStoreIndex.from_documents(documents)
        return index

# --- CHAT INTERACTION ---
if prompt := st.chat_input("Ask a question about your document..."):
    # User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="https://cdn-icons-png.flaticon.com/512/9131/9131529.png"):
        st.markdown(prompt)

    # Validation
    if not uploaded_file:
        st.warning("⚠️ Please upload a PDF document in the sidebar to proceed.")
        st.stop()

    # AI Processing
    with st.chat_message("assistant", avatar="https://cdn-icons-png.flaticon.com/512/4712/4712027.png"):
        with st.spinner("Processing inquiry..."):
            llm, embed_model, error_msg = load_models(api_key)
            
            if error_msg:
                st.error(f"System Error: {error_msg}")
                st.stop()
            
            Settings.llm = llm
            Settings.embed_model = embed_model

            try:
                # Index only if new session or file changed (basic check)
                if "vector_index" not in st.session_state:
                     with st.spinner("Indexing knowledge base..."):
                        st.session_state.vector_index = index_document(uploaded_file)
                
                query_engine = st.session_state.vector_index.as_query_engine()
                response = query_engine.query(prompt)
                
                st.markdown(response.response)
                st.session_state.messages.append({"role": "assistant", "content": response.response})
                
            except Exception as e:
                st.error(f"Processing Error: {str(e)}")