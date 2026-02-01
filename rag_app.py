import streamlit as st
import tempfile
import os

# --- Imports ---
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import InferenceClient

# Document Import
from langchain_core.documents import Document

# OCR Imports
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="Personal Knowledge Assistant",
    page_icon="🧠",
    layout="wide"
)

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Setup")
    
    hf_api_token = st.text_input(
        "Hugging Face API Token",
        type="password",
        help="Get your free token from https://huggingface.co/settings/tokens"
    )

    # Add a connection test button
    if hf_api_token:
        if st.button("🔌 Test Connection"):
            try:
                client = InferenceClient(token=hf_api_token)
                # Simple test
                client.text_generation("Hello", model="google/flan-t5-large")
                st.success("✅ Connected successfully!")
            except Exception as e:
                st.error(f"❌ Connection failed: {e}")
    
    # Recommended model first
    model_id = st.selectbox(
        "Select LLM",
        [
            "google/flan-t5-large", # Most reliable free model
            "mistralai/Mistral-7B-Instruct-v0.3", # Good but needs license
            "tiiuae/falcon-7b-instruct"
        ],
        index=0
    )
    
    st.divider()
    
    uploaded_files = st.file_uploader(
        "Upload Documents (PDF, TXT)", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    
    process_btn = st.button("Ingest Documents")

# --- Logic ---

def get_llm_response(prompt, context, api_key, repo_id):
    client = InferenceClient(token=api_key)
    
    # Simplified prompt to save tokens
    full_prompt = f"""
    Use the Context below to answer the Question. If the answer isn't there, say "I don't know".
    
    Context:
    {context}
    
    Question: {prompt}
    
    Answer:
    """
    
    try:
        response = client.text_generation(
            prompt=full_prompt,
            model=repo_id,
            max_new_tokens=250, # Reduced to prevent timeouts
            temperature=0.1,
            return_full_text=False
        )
        return response
    except Exception as e:
        # RETURN THE ACTUAL ERROR for debugging
        return f"ERROR_DETAILS: {str(e)}"

def ocr_pdf(file_path):
    text_content = ""
    try:
        images = convert_from_path(file_path)
        for image in images:
            page_text = pytesseract.image_to_string(image)
            text_content += page_text + "\n"
    except Exception as e:
        return ""
    return text_content

def process_documents(files):
    documents = []
    status_text = st.empty()
    status_text.info("Reading files...")
    
    for uploaded_file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
            
        try:
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load()
                
                # Check for scan
                total_len = sum([len(d.page_content.strip()) for d in docs])
                if total_len < 50: 
                    st.warning(f"⚠️ Scanned PDF detected: {uploaded_file.name}. Running OCR...")
                    ocr_text = ocr_pdf(tmp_file_path)
                    if ocr_text.strip():
                        docs = [Document(page_content=ocr_text, metadata={"source": uploaded_file.name, "type": "ocr"})]
            else:
                loader = TextLoader(tmp_file_path)
                docs = loader.load()
                
            for doc in docs:
                if 'source' not in doc.metadata:
                    doc.metadata['source'] = uploaded_file.name
                
            documents.extend(docs)
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
            
    if not documents:
        return None

    status_text.info("Chunking...")
    
    # Smaller chunks for reliability
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    splits = text_splitter.split_documents(documents)
    
    if not splits:
        st.error("No text found.")
        return None
    
    status_text.info("Embedding...")
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(splits, embeddings)
        status_text.success("Ready!")
        return vector_store
    except Exception as e:
        st.error(f"Embedding Error: {e}")
        return None

# --- UI ---

st.header("🧠 Personal Knowledge Assistant")

if process_btn and uploaded_files:
    with st.spinner("Processing..."):
        st.session_state.vector_store = process_documents(uploaded_files)

if st.session_state.vector_store:
    # Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("Sources"):
                    for s in message["sources"]:
                        st.caption(f"{s['source']}: {s['content']}")

    # Input
    if prompt := st.chat_input("Ask a question..."):
        if not hf_api_token:
            st.error("❌ Key missing!")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Retrieve only TOP 2 to save space
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 2})
                    relevant_docs = retriever.invoke(prompt)
                    
                    context_text = "\n\n".join([d.page_content for d in relevant_docs])
                    
                    response_text = get_llm_response(prompt, context_text, hf_api_token, model_id)
                    
                    # Error Handling in UI
                    if "ERROR_DETAILS" in response_text:
                        st.error("API Error encountered:")
                        st.code(response_text)
                        st.info("Tip: Try switching to 'google/flan-t5-large' in the sidebar.")
                    else:
                        st.markdown(response_text)
                        
                        sources_data = [
                            {"source": doc.metadata.get('source', 'Unknown'), "content": doc.page_content[:200] + "..."}
                            for doc in relevant_docs
                        ]
                        
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response_text,
                            "sources": sources_data
                        })
                        
                        with st.expander("Sources"):
                            for s in sources_data:
                                st.caption(f"{s['source']}: {s['content']}")
