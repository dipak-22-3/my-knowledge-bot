import streamlit as st
import tempfile
import os

# --- Imports ---
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import InferenceClient

# FIX: Updated import for Document
from langchain_core.documents import Document

# OCR Imports
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="Personal Knowledge Assistant (+OCR)",
    page_icon="🧠",
    layout="wide"
)

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- Sidebar: Configuration & Upload ---
with st.sidebar:
    st.title("⚙️ Setup")
    
    # API Key Input
    hf_api_token = st.text_input(
        "Hugging Face API Token",
        type="password",
        help="Get your free token from https://huggingface.co/settings/tokens"
    )
    
    # Model Selection
    model_id = st.selectbox(
        "Select LLM",
        ["mistralai/Mistral-7B-Instruct-v0.3", "tiiuae/falcon-7b-instruct", "google/flan-t5-large"],
        index=0
    )
    
    st.divider()
    
    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload Documents (PDF, TXT)", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    
    process_btn = st.button("Ingest Documents")

# --- Helper Functions ---

def get_llm_response(prompt, context, api_key, repo_id):
    """
    Query the Hugging Face Inference API.
    """
    client = InferenceClient(token=api_key)
    
    full_prompt = f"""
    You are a helpful Knowledge Assistant. Use the provided context to answer the question.
    If the answer is not in the context, say "I couldn't find a reliable source for that in your documents."
    
    CONTEXT:
    {context}
    
    QUESTION:
    {prompt}
    
    ANSWER (Cite specific documents if possible):
    """
    
    try:
        response = client.text_generation(
            prompt=full_prompt,
            model=repo_id,
            max_new_tokens=512,
            temperature=0.1,
            return_full_text=False
        )
        return response
    except Exception as e:
        return f"Error contacting API: {str(e)}"

def ocr_pdf(file_path):
    """
    Fallback function: Converts PDF pages to images and extracts text using Tesseract.
    """
    text_content = ""
    try:
        # Convert PDF to images
        images = convert_from_path(file_path)
        for i, image in enumerate(images):
            # Extract text from image
            page_text = pytesseract.image_to_string(image)
            text_content += page_text + "\n"
    except Exception as e:
        st.error(f"OCR Error: {e}")
        return ""
    return text_content

def process_documents(files):
    """
    Load, chunk, and embed documents (with OCR fallback).
    """
    documents = []
    status_text = st.empty()
    status_text.info("Reading files...")
    
    for uploaded_file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
            
        try:
            # 1. Try Standard Loading First (Fast)
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load()
                
                # Check if standard loading returned empty text
                total_text_length = sum([len(d.page_content.strip()) for d in docs])
                
                # 2. If text is empty/very short, trigger OCR (Slow but effective)
                if total_text_length < 50: 
                    st.warning(f"⚠️ '{uploaded_file.name}' seems to be a scanned image. Starting OCR (this allows reading images but takes longer)...")
                    ocr_text = ocr_pdf(tmp_file_path)
                    if ocr_text.strip():
                        # Create a new Document object from OCR text
                        docs = [Document(page_content=ocr_text, metadata={"source": uploaded_file.name, "type": "ocr"})]
                    else:
                        st.error(f"Failed to extract text from {uploaded_file.name} even with OCR.")
            else:
                loader = TextLoader(tmp_file_path)
                docs = loader.load()
                
            # Add metadata
            for doc in docs:
                if 'source' not in doc.metadata:
                    doc.metadata['source'] = uploaded_file.name
                
            documents.extend(docs)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
            
    if not documents:
        return None

    status_text.info("Chunking text...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    
    if not splits:
        st.error("Documents were empty after processing.")
        return None
    
    status_text.info(f"Generating embeddings for {len(splits)} chunks...")
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(splits, embeddings)
        status_text.success("Ingestion Complete! You can now chat.")
        return vector_store
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return None

# --- Main Logic ---

st.header("🧠 Personal Knowledge Assistant (+OCR)")
st.caption("Ask questions based on your uploaded documents (Scans supported).")

if process_btn and uploaded_files:
    with st.spinner("Processing knowledge base..."):
        st.session_state.vector_store = process_documents(uploaded_files)

if st.session_state.vector_store is None:
    st.info("👈 Please upload documents and click 'Ingest Documents' to start.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📚 View Sources"):
                    for idx, source in enumerate(message["sources"]):
                        st.markdown(f"**Source {idx+1} ({source['source']}):**")
                        st.caption(source['content'])

    if prompt := st.chat_input("Ask a question about your documents..."):
        if not hf_api_token:
            st.error("Please enter a Hugging Face API Token in the sidebar.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving context & generating answer..."):
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                    relevant_docs = retriever.invoke(prompt)
                    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
                    response_text = get_llm_response(prompt, context_text, hf_api_token, model_id)
                    
                    st.markdown(response_text)
                    
                    sources_data = [
                        {"source": doc.metadata.get('source', 'Unknown'), "content": doc.page_content[:300] + "..."}
                        for doc in relevant_docs
                    ]
                    
                    with st.expander("📚 View Sources & Confidence"):
                        for idx, doc in enumerate(relevant_docs):
                            src_label = doc.metadata.get('source', 'Unknown')
                            if doc.metadata.get('type') == 'ocr':
                                src_label += " (OCR Scanned)"
                            st.markdown(f"**Source {idx+1}**: *{src_label}*")
                            st.caption(doc.page_content)

            st.session_state.messages.append({
                "role": "assistant", 
                "content": response_text,
                "sources": sources_data
            })
