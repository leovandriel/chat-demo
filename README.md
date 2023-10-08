Chat Demo
=========

This is a demo for a chat agent that answers questions based on documents, using document embeddings (Chroma) and an LLM (LangChain, OpenAI).

# Usage

Create `openai_api_key.txt` in project root with your [OpenAI API key](https://platform.openai.com/account/api-keys). To limit access:

    chmod 600 openai_api_key.txt

Install requirements:

    pip install -r requirements.txt

Populate the document list in `documents.txt`. For example, to get urls of www.example.com:

    wget -m www.example.com 2>&1 | grep -o "www.example.com/.*"

Load documents listed in `documents.txt` and store embeddings in a Chroma vector store:

    python src/load.py

To query the documents in CLI:

    python src/ask.py

To run local Streamlit app:

    streamlit run src/app.py

# License

MIT
