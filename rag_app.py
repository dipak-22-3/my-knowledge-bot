import streamlit as st
import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile

# --- Page Config ---
st.set_page_config(page_title="Gemini RAG Bot", page_icon="🤖")

# --- 1. API Key Setup (Automatic) ---
# Yeh line automatically Streamlit Secrets se key uthayegi
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
except Exception:
    st.error("🚨 API Key nahi mili! Streamlit Settings > Secrets mein 'GOOGLE_API_KEY' add karein.")
    st.stop()

# --- Session State ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Helper Functions ---
def get_vector_store(uploaded_files):
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    with st.status("Processing documents...", expanded=True) as status:
        for uploaded_file in uploaded_files:
            # Temp file banakar read karo
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                if uploaded_file.name.endswith(".pdf"):
                    loader = PyPDFLoader(tmp_path)
                else:
                    loader = TextLoader(tmp_path)
                
                docs = loader.load()
                # Source metadata add karo
                for doc in docs:
                    doc.metadata["source"] = uploaded_file.name
                
                documents.extend(docs)
            except Exception as e:
                st.error(f"Error reading {uploaded_file.name}: {e}")
            finally:
                os.remove(tmp_path)

        if not documents:
            return None

        # Chunks banao
        splits = text_splitter.split_documents(documents)
        
        # Google Embeddings (Free & Fast)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        
        # Vector DB create karo
        vector_store = FAISS.from_documents(splits, embeddings)
        status.update(label="✅ Knowledge Base Ready!", state="complete", expanded=False)
        return vector_store

def get_gemini_response(question, vector_store):
    # 1. Context dhoondo
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # 2. Gemini se poocho
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
    
    prompt = f"""You are a helpful assistant. Use the Context below to answer the user's question.
    If the answer is not in the context, say "I don't know based on this document."
    
    Context:
    {context}
    
    Question: {question}
    """
    
    response = llm.invoke(prompt)
    return response.content, relevant_docs

# --- UI Layout ---
st.title("🤖 Personal Knowledge Bot (Gemini)")
st.caption("Powered by Google Gemini - Fast & Free")

with st.sidebar:
    st.header("📂 Data Upload")
    uploaded_files = st.file_uploader("Upload PDF/TXT", accept_multiple_files=True)
    
    if st.button("Submit & Process"):
        if uploaded_files:
            st.session_state.vector_store = get_vector_store(uploaded_files)
        else:
            st.warning("Pehle file upload karo!")

# --- Chat Interface ---
if st.session_state.vector_store:
    # Chat History dikhao
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Naya sawal
    if prompt := st.chat_input("Ask something about your file..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = get_gemini_response(prompt, st.session_state.vector_store)
                st.markdown(answer)
                
                # Sources dikhao
                with st.expander("View Sources"):
                    for doc in sources:
                        st.caption(f"📄 {doc.metadata['source']}: {doc.page_content[:100]}...")
        
        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("👈 Sidebar mein Document upload karo aur 'Submit' dabao.")
