import streamlit as st
import os
import tempfile
import dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# 1) MUST be the very first Streamlit command:
st.set_page_config(page_title="PDF Chatbot", layout="centered")

# Load environment variables
dotenv.load_dotenv()

# UI Header (now comes after set_page_config)
st.title("💬 RAG Chatbot")
st.write("Upload a PDF and ask questions about its content.")

# 2) PDF Upload widget
uploaded_file = st.file_uploader("📄 Upload a PDF", type=["pdf"])

# 3) User query input
query = st.chat_input("Ask something about the PDF...")

# 4) Embedding model setup
EMB_MODEL = os.getenv("EMBEDDING_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
embedding = HuggingFaceEmbeddings(
    model_name=EMB_MODEL,
    model_kwargs={"device": "cpu"}  # force CPU for cloud
)

# 5) Process uploaded PDF
if uploaded_file is not None:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    # Load and split PDF into documents
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = splitter.split_documents(pages)

    # Build an in-memory Chroma vector store
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        collection_name="pdf_docs"  # no disk persistence
    )
    retriever = vectordb.as_retriever()

    # Initialize the LLM and QA chain
    llm = Ollama(model="tinyllama")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # 6) Respond to the query
    if query:
        with st.spinner("Thinking..."):
            answer = qa_chain.run(query)
        st.chat_message("user").write(query)
        st.chat_message("assistant").write(answer)

else:
    st.warning("👆 Please upload a PDF to start chatting.")
