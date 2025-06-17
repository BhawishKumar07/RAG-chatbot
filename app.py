import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import dotenv

# Load environment variables (if any)
dotenv.load_dotenv()

st.set_page_config(page_title="PDF Chatbot", layout="centered")
st.title("💬 RAG Chatbot")
st.write("Upload your PDF and ask questions about it!")

uploaded_file = st.file_uploader("📄 Upload PDF", type=["pdf"])
query = st.chat_input("Ask something about the PDF...")

# ✅ Use sentencepiece-free embedding model
EMB_MODEL = "intfloat/e5-small-v2"

# Initialize embeddings (safe on CPU)
embedding = HuggingFaceEmbeddings(
    model_name=EMB_MODEL,
    model_kwargs={"device": "cpu"}
)

# Process PDF if uploaded
if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    documents = text_splitter.split_documents(docs)

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        collection_name="pdf_docs"
    )

    retriever = vectordb.as_retriever()
    llm = Ollama(model="tinyllama")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    if query:
        with st.spinner("Thinking..."):
            result = qa_chain.run(query)
        st.chat_message("user").write(query)
        st.chat_message("assistant").write(result)
else:
    st.warning("Please upload a PDF to continue.")
