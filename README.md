Chat Demo
=========

This is a demo for a chat agent that answers questions based on documents, using document embeddings (Chroma) and an LLM (LangChain), and supporting multiple LLM APIs (OpenAI, Cohere, Anthropic, Google Palm, VertexAI)

# Usage

Install requirements:

    pip install -r requirements.txt

Depending on the LLM API you use, store API keys and limit access:

    chmod 600 openai_api_key.txt
    chmod 600 cohere_api_key.txt
    chmod 600 anthropic_api_key.txt
    chmod 600 google_api_key.txt
    chmod 600 vertex_api_key.txt

Populate the document list in `documents.txt`. For example, to get urls of www.example.com:

    wget -m www.example.com 2>&1 | grep -o "www.example.com/.*"

Load documents listed in `documents.txt` and store embeddings in a Chroma vector store:

    python load.py

To query the documents in CLI:

    python ask.py

To run local Streamlit app:

    streamlit run app.py

To run evals:

    PYTHONPATH=. OPENAI_API_KEY=$(< openai_api_key.txt) oaieval chat_completion_fn,gpt-4 advice --registry_path ./eval

# License

MIT
