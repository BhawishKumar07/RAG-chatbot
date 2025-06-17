# ingest.py

import os
import dotenv
from typing import List
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

# Load environment variables
dotenv.load_dotenv()
HF_TOKEN = os.getenv("HF_API_TOKEN")
EMB_MODEL = os.getenv("EMBEDDING_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"

# 1) PDF → Text Loader
def load_pdfs(folder: str = "data") -> List[tuple]:
    docs = []
    for fn in os.listdir(folder):
        if fn.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder, fn))
            text = "".join([p.extract_text() or "" for p in reader.pages])
            docs.append((fn, text))
    return docs

# 2) Text Splitter
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# 3) Wrapper for Chroma-compatible Embedding
class ChromaCompatibleEmbeddings:
    def __init__(self, model_name: str):
        self.model = HuggingFaceEmbeddings(model_name=model_name)

    # Must use "input" instead of "texts" for Chroma compatibility
    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.model.embed_documents(input)

# 4) Embeddings and Chroma setup
embeddings = ChromaCompatibleEmbeddings(model_name=EMB_MODEL)
client = chromadb.Client()
collection = client.create_collection(name="pdf_docs", embedding_function=embeddings)

# 5) Ingestion Function
def ingest():
    docs = load_pdfs()
    for fn, text in docs:
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            metadata = {"source": fn, "chunk": i}
            doc_id = f"{fn}-{i}"
            collection.add(documents=[chunk], metadatas=[metadata], ids=[doc_id])
    print("Ingestion complete. Total documents:", collection.count())

# 6) Main
if __name__ == "__main__":
    ingest()
