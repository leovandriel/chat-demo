"""Load documents from the web and create a vectorstore."""

import logging
from pathlib import Path

import tiktoken
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from .config import data_dir
from .model import create_embedder

chroma_file = data_dir / "chroma.sqlite3"
min_page_content_length = 10

logging.basicConfig(level=logging.INFO)


def load() -> None:
    """Load documents from the web and create a vectorstore."""
    with Path("documents.txt").open() as f:
        urls = f.read().split("\n")
    urls = [url for url in urls if url != ""]

    logging.info("loading %i documents", len(urls))
    loader = WebBaseLoader(urls)
    docs = loader.load()
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = sum(len(enc.encode(doc.page_content)) for doc in docs)
    characters = sum(len(doc.page_content) for doc in docs)
    for doc in docs:
        if len(doc.page_content) < min_page_content_length:
            msg = f"Empty doc: {doc}"
            raise RuntimeError(msg)

    logging.info("splitting %i tokens (%i chars)", tokens, characters)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    logging.info("embedding %i chunks", len(chunks))
    embedder = create_embedder(vendor="openai", model="text-embedding-ada-002")
    if chroma_file.exists():
        chroma_file.unlink()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=data_dir.as_posix(),
    )

    logging.info("saving to %s", chroma_file)
    vectorstore.persist()
