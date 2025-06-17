# app.py
import os
import dotenv
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings

# Load env
dotenv.load_dotenv()
EMB_MODEL = os.getenv("EMBEDDING_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"

# Load embedding model
embedding = HuggingFaceEmbeddings(model_name=EMB_MODEL)

# Load vector database
vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding,
    collection_name="pdf_docs"
)
retriever = vectordb.as_retriever()

# Load LLM (Ollama - llama3)
llm = Ollama(model="llama3")

# Build RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# Streamlit UI
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("📄 RAG PDF Chatbot")
st.write("Ask anything from your PDF files.")

query = st.text_input("Ask a question:")

if query:
    with st.spinner("Thinking..."):
        answer = qa_chain.run(query)
    st.markdown(f"**Answer:** {answer}")