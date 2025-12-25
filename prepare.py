import glob

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings


pdf_paths = glob.glob("data/Everstorm_*.pdf")
raw_docs = []

print(f"Loaded {len(raw_docs)} PDF pages from {len(pdf_paths)} files.")


for pdf_path in pdf_paths:
    loader = PyPDFLoader(pdf_path)
    raw_docs.extend(loader.load())

chunks = []

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = text_splitter.split_documents(raw_docs)
print(f"✅ {len(chunks)} chunks ready for embedding")



embeddings = SentenceTransformerEmbeddings(model_name="thenlper/gte-small")

vectordb = FAISS.from_documents(chunks, embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 8})

vectordb.save_local("faiss_index")

print("✅ Vector store with", vectordb.index.ntotal, "embeddings")