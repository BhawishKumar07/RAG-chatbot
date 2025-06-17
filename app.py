import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import os
import dotenv

# Load environment variables
dotenv.load_dotenv()

# UI Header
st.set_page_config(page_title="PDF Chatbot", layout="centered")
st.title("💬 RAG Chatbot")
st.write("Ask questions based on your PDF content.")

# Input box
query = st.chat_input("Ask something about your PDF...")

# Initialize Vector Store & Model
EMB_MODEL = os.getenv("EMBEDDING_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"

# Updated embedding line using safer class
embedding = SentenceTransformerEmbeddings(model_name=EMB_MODEL)

vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding,
    collection_name="pdf_docs"
)
retriever = vectordb.as_retriever()

llm = Ollama(model="tinyllama")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Show Response
if query:
    with st.spinner("Thinking..."):
        result = qa_chain.run(query)
    st.chat_message("user").write(query)
    st.chat_message("assistant").write(result)
