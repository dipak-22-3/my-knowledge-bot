import streamlit as st
import tempfile
import os

# Imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import InferenceClient

# --- Page Configuration ---
st.set_page_config(
    page_title="Personal Knowledge Assistant",
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
    
    # Construct a structured prompt for RAG
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
            temperature=0.1, # Low temperature for factual grounding
            return_full_text=False
        )
        return response
    except Exception as e:
        return f"Error contacting API: {str(e)}"

def process_documents(files):
    """
    Load, chunk, and embed documents.
    """
    documents = []
    
    status_text = st.empty()
    status_text.info("Reading files...")
    
    for uploaded_file in files:
        # Create a temp file to load data
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
            
        try:
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load()
            else:
                loader = TextLoader(tmp_file_path)
                docs = loader.load()
                
            # Add metadata for citations
            for doc in docs:
                doc.metadata['source'] = uploaded_file.name
                
            documents.extend(docs)
        except Exception as e:
            st.error(f"Error reading {uploaded_file.name}: {e}")
        finally:
            os.remove(tmp_file_path)
            
    # SAFETY CHECK 1: Did we load anything?
    if not documents:
        st.warning("No documents loaded.")
        return None

    # SAFETY CHECK 2: Do the documents actually contain text?
    # This filters out empty pages or scanned images without OCR
    documents = [doc for doc in documents if doc.page_content.strip()]
    
    if not documents:
        st.error("⚠️ No text could be extracted. If you uploaded a PDF, it might be a scanned image (picture) which cannot be read without OCR.")
        return None

    status_text.info("Chunking text...")
    
    # Chunking Strategy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    
    # SAFETY CHECK 3: Did chunking result in actual splits?
    if not splits:
        st.error("Documents were empty after processing.")
        return None
    
    status_text.info(f"Generating embeddings for {len(splits)} chunks... (This may take a moment)")
    
    try:
        # Embeddings - Using a lightweight, high-performance model
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Vector Store - FAISS
        vector_store = FAISS.from_documents(splits, embeddings)
        
        status_text.success("Ingestion Complete! You can now chat.")
        return vector_store
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return None

# --- Main Logic ---

st.header("🧠 Personal Knowledge Assistant (RAG)")
st.caption("Ask questions based on your uploaded documents.")

# Process files if button clicked
if process_btn and uploaded_files:
    with st.spinner("Processing knowledge base..."):
        st.session_state.vector_store = process_documents(uploaded_files)

# Warning if no DB
if st.session_state.vector_store is None:
    st.info("👈 Please upload documents and click 'Ingest Documents' to start.")
else:
    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📚 View Sources"):
                    for idx, source in enumerate(message["sources"]):
                        st.markdown(f"**Source {idx+1} ({source['source']}):**")
                        st.caption(source['content'])

    # Chat Input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not hf_api_token:
            st.error("Please enter a Hugging Face API Token in the sidebar.")
        else:
            # User Message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Assistant Logic
            with st.chat_message("assistant"):
                with st.spinner("Retrieving context & generating answer..."):
                    
                    # 1. Retrieval (Top k=3)
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                    relevant_docs = retriever.invoke(prompt)
                    
                    # Prepare Context text
                    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
                    
                    # 2. Generation
                    response_text = get_llm_response(prompt, context_text, hf_api_token, model_id)
                    
                    st.markdown(response_text)
                    
                    # Prepare sources for storage/display
                    sources_data = [
                        {"source": doc.metadata.get('source', 'Unknown'), "content": doc.page_content[:300] + "..."}
                        for doc in relevant_docs
                    ]
                    
                    with st.expander("📚 View Sources & Confidence"):
                        for idx, doc in enumerate(relevant_docs):
                            st.markdown(f"**Source {idx+1}**: *{doc.metadata.get('source', 'Unknown')}*")
                            st.caption(doc.page_content)

            # Update History
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response_text,
                "sources": sources_data
            })
