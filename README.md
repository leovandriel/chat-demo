Chat Demo
=========

This is a demo for a chat agent that answers questions based on documents, using document embeddings (Chroma) and an LLM (LangChain), and supporting multiple LLM APIs (OpenAI, Cohere)

# Usage

Install requirements:

    pip install -r requirements.txt

Depending on the LLM API you use, store API keys and limit access:

    chmod 600 openai_api_key.txt
    chmod 600 cohere_api_key.txt
    chmod 600 anthropic_api_key.txt

Populate the document list in `documents.txt`. For example, to get urls of www.example.com:

    wget -m www.example.com 2>&1 | grep -o "www.example.com/.*"

Load documents listed in `documents.txt` and store embeddings in a Chroma vector store:

    python src/load.py

To query the documents in CLI:

    python src/ask.py

To run local Streamlit app:

    streamlit run src/app.py

To export vector store:

    sqlite3 data/chroma.sqlite3 .dump > vector.sql

To import vector store:

    rm data/chroma.sqlite3
    cat vector.sql | sqlite3 data/chroma.sqlite3

# License

MIT
