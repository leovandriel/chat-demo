import os
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
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    documents = splitter.split_documents(data)

    embedder = create_embedder()

    if os.path.exists(chroma_file):
        os.remove(chroma_file)
    vectorstore = Chroma.from_documents(
        documents=documents, embedding=embedder, persist_directory=data_dir
    )
    vectorstore.persist()


if __name__ == "__main__":
    load()
