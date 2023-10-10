import json
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import (
    ChatOpenAI,
    ChatCohere,
    ChatAnthropic,
    ChatGooglePalm,
    ChatVertexAI,
)
from langchain.embeddings import (
    OpenAIEmbeddings,
    CohereEmbeddings,
    GooglePalmEmbeddings,
    VertexAIEmbeddings,
)
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from config import (
    data_dir,
    chat_llm_vendor,
    embed_llm_vendor,
    chat_model_name,
    embed_model_name,
    agent_skill,
    prompt_template,
    document_template,
)
from util import get_secret


def create_llm(streaming=False, handler=None):
    if chat_llm_vendor == "openai":
        return ChatOpenAI(
            model_name=chat_model_name,
            openai_api_key=get_secret("openai_api_key"),
            streaming=streaming,
            callbacks=[handler] if handler else [],
        )
    elif chat_llm_vendor == "cohere":
        return ChatCohere(
            model_name=chat_model_name,
            cohere_api_key=get_secret("cohere_api_key"),
            streaming=streaming,
            callbacks=[handler] if handler else [],
        )
    elif chat_llm_vendor == "anthropic":
        return ChatAnthropic(
            model=chat_model_name,
            anthropic_api_key=get_secret("anthropic_api_key"),
            streaming=streaming,
            callbacks=[handler] if handler else [],
        )
    elif chat_llm_vendor == "google":
        return ChatGooglePalm(
            model=chat_model_name,
            google_api_key=get_secret("google_api_key"),
            streaming=streaming,
            callbacks=[handler] if handler else [],
        )
    elif chat_llm_vendor == "vertex":
        from google.oauth2.service_account import Credentials

        info = json.loads(get_secret("vertex_api_key"))
        return ChatVertexAI(
            model=chat_model_name,
            credentials=Credentials.from_service_account_info(info),
            project=info["project_id"],
            streaming=streaming,
            callbacks=[handler] if handler else [],
        )


def create_embedder():
    if embed_llm_vendor == "openai":
        return OpenAIEmbeddings(
            model=embed_model_name,
            openai_api_key=get_secret("openai_api_key"),
        )
    elif embed_llm_vendor == "cohere":
        return CohereEmbeddings(
            model=embed_model_name,
            cohere_api_key=get_secret("cohere_api_key"),
        )
    elif embed_llm_vendor == "google":
        return GooglePalmEmbeddings(
            model_name=embed_model_name,
            google_api_key=get_secret("google_api_key"),
        )
    elif embed_llm_vendor == "vertex":
        from google.oauth2.service_account import Credentials

        info = json.loads(get_secret("vertex_api_key"))
        return VertexAIEmbeddings(
            model_name=embed_model_name,
            credentials=Credentials.from_service_account_info(info),
            project=info["project_id"],
        )


def load_store():
    embedder = create_embedder()
    vectorstore = Chroma(
        embedding_function=embedder,
        persist_directory=data_dir,
    )
    return vectorstore


class StreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.callback = None

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.callback(token)


def setup_chain(vectorstore, streaming):
    retriever = vectorstore.as_retriever()

    internal_llm = create_llm()

    memory = ConversationSummaryBufferMemory(
        llm=internal_llm,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    prompt = PromptTemplate.from_template("\n".join([agent_skill, prompt_template]))
    document_prompt = PromptTemplate.from_template(document_template)
    handler = StreamHandler()

    response_llm = create_llm(streaming=streaming, handler=handler)

    chain = ConversationalRetrievalChain.from_llm(
        response_llm,
        condense_question_llm=internal_llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={
            "prompt": prompt,
            "document_prompt": document_prompt,
        },
    )

    def run(question, callback=None):
        handler.callback = callback
        return chain({"question": question})["answer"]

    return run
