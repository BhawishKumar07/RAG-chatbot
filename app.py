import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import dotenv
from pathlib import Path

# Load environment variables
dotenv.load_dotenv()

# UI Header
st.set_page_config(page_title="PDF Chatbot", layout="centered")
st.title("💬 RAG Chatbot")
st.write("Ask questions based on your PDF content.")

# Input box
query = st.chat_input("Ask something about your PDF...")

# Set embedding model
EMB_MODEL = os.getenv("EMBEDDING_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
embedding = HuggingFaceEmbeddings(
    model_name=EMB_MODEL,
    model_kwargs={"device": "cpu"}  # use CPU to avoid GPU issues
)

# Load sample text data (replace with your PDF document parsing)
docs_folder = "./docs"  # make sure you upload your PDFs/texts to this folder
if not os.path.exists(docs_folder):
    os.makedirs(docs_folder)
    # Create a dummy file to avoid crash
    with open(os.path.join(docs_folder, "example.txt"), "w") as f:
        f.write("This is an example document. Replace it with actual PDF loading.")

# Load and split documents
loader = TextLoader(os.path.join(docs_folder, "example.txt"))  # replace this with PDFLoader if needed
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = splitter.split_documents(documents)

# Create FAISS vectorstore
vectordb = FAISS.from_documents(texts, embedding)
retriever = vectordb.as_retriever()

# Set LLM
llm = Ollama(model="tinyllama")

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Show response
if query:
    with st.spinner("Thinking..."):
        result = qa_chain.run(query)
    st.chat_message("user").write(query)
    st.chat_message("assistant").write(result)
