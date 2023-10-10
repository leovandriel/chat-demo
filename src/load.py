import os
import tiktoken
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from config import data_dir
from model import create_embedder

chroma_file = f"{data_dir}/chroma.sqlite3"


def load():
    with open("documents.txt", "r") as f:
        urls = f.read().split("\n")
    urls = [url for url in urls if url != ""]

    print(f"loading {len(urls)} documents")
    loader = WebBaseLoader(urls)
    data = loader.load()
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = sum(len(enc.encode(doc.page_content)) for doc in data)
    characters = sum(len(doc.page_content) for doc in data)

    print(f"splitting {tokens} tokens ({characters} chars)")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    documents = splitter.split_documents(data)

    print(f"embedding {len(documents)} chunks")
    embedder = create_embedder(vendor="openai", model="text-embedding-ada-002")
    if os.path.exists(chroma_file):
        os.remove(chroma_file)
    vectorstore = Chroma.from_documents(
        documents=documents, embedding=embedder, persist_directory=data_dir
    )

    print(f"saving to {chroma_file}")
    vectorstore.persist()


if __name__ == "__main__":
    load()
