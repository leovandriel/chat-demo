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
    agent_skill,
    prompt_template,
    document_template,
)
from util import get_secret


def create_llm(vendor, model, temperature=0.5, streaming=False, handler=None):
    if vendor == "openai":
        return ChatOpenAI(
            model_name=model,
            temperature=temperature,
            openai_api_key=get_secret("openai_api_key"),
            streaming=streaming,
            callbacks=[handler] if handler else [],
        )
    elif vendor == "cohere":
        return ChatCohere(
            model_name=model,
            temperature=temperature,
            cohere_api_key=get_secret("cohere_api_key"),
            streaming=streaming,
            callbacks=[handler] if handler else [],
        )
    elif vendor == "anthropic":
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            anthropic_api_key=get_secret("anthropic_api_key"),
            streaming=streaming,
            callbacks=[handler] if handler else [],
        )
    elif vendor == "google":
        return ChatGooglePalm(
            model=model,
            temperature=temperature,
            google_api_key=get_secret("google_api_key"),
            streaming=streaming,
            callbacks=[handler] if handler else [],
        )
    elif vendor == "vertex":
        from google.oauth2.service_account import Credentials

        info = json.loads(get_secret("vertex_api_key"))
        return ChatVertexAI(
            model=model,
            temperature=temperature,
            credentials=Credentials.from_service_account_info(info),
            project=info["project_id"],
            streaming=streaming,
            callbacks=[handler] if handler else [],
        )


def create_embedder(vendor, model):
    if vendor == "openai":
        return OpenAIEmbeddings(
            model=model,
            openai_api_key=get_secret("openai_api_key"),
        )
    elif vendor == "cohere":
        return CohereEmbeddings(
            model=model,
            cohere_api_key=get_secret("cohere_api_key"),
        )
    elif vendor == "google":
        return GooglePalmEmbeddings(
            model_name=model,
            google_api_key=get_secret("google_api_key"),
        )
    elif vendor == "vertex":
        from google.oauth2.service_account import Credentials

        info = json.loads(get_secret("vertex_api_key"))
        return VertexAIEmbeddings(
            model_name=model,
            credentials=Credentials.from_service_account_info(info),
            project=info["project_id"],
        )


def load_store():
    embedder = create_embedder(vendor="openai", model="text-embedding-ada-002")
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

    internal_llm = create_llm(vendor="openai", model="gpt-3.5-turbo", temperature=0.2)

    memory = ConversationSummaryBufferMemory(
        llm=internal_llm,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    prompt = PromptTemplate.from_template("\n".join([agent_skill, prompt_template]))
    document_prompt = PromptTemplate.from_template(document_template)
    handler = StreamHandler()

    response_llm = create_llm(
        vendor="openai",
        model="gpt-3.5-turbo",
        temperature=0.5,
        streaming=streaming,
        handler=handler,
    )

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
