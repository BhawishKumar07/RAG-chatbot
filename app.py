import streamlit as st
import os
import tempfile
import dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
dotenv.load_dotenv()

# Streamlit UI setup
st.set_page_config(page_title="PDF Chatbot", layout="centered")
st.title("💬 RAG Chatbot")
st.write("Upload a PDF and ask questions based on its content.")

# Upload PDF
uploaded_file = st.file_uploader("📄 Upload a PDF", type=["pdf"])

# User query input
query = st.chat_input("Ask something about your PDF...")

# Initialize model
EMB_MODEL = os.getenv("EMBEDDING_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"

# Set up embedding function
embedding = HuggingFaceEmbeddings(
    model_name=EMB_MODEL,
    model_kwargs={"device": "cpu"}
)

# If PDF is uploaded
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # Load and split PDF into chunks
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(pages)

    # Store vectors
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory="./chroma_db",
        collection_name="pdf_docs"
    )
    vectordb.persist()
    retriever = vectordb.as_retriever()

    # Initialize LLM + Chain
    llm = Ollama(model="tinyllama")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    if query:
        with st.spinner("Thinking..."):
            response = qa_chain.run(query)
        st.chat_message("user").write(query)
        st.chat_message("assistant").write(response)
else:
    st.warning("👆 Please upload a PDF to begin.")
