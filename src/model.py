from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI, ChatCohere
from langchain.embeddings import OpenAIEmbeddings, CohereEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from config import (
    data_dir,
    llm_vendor,
    model_name,
    agent_skill,
    prompt_template,
    document_template,
)
from util import get_secret


def create_llm(streaming=False, handler=None):
    if llm_vendor == "openai":
        return ChatOpenAI(
            model_name=model_name,
            temperature=0,
            openai_api_key=get_secret("openai_api_key"),
            streaming=streaming,
            callbacks=[handler] if handler else [],
        )
    elif llm_vendor == "cohere":
        return ChatCohere(
            model_name=model_name,
            temperature=0,
            cohere_api_key=get_secret("cohere_api_key"),
            streaming=streaming,
            callbacks=[handler] if handler else [],
        )


def create_embedder():
    if llm_vendor == "openai":
        return OpenAIEmbeddings(openai_api_key=get_secret("openai_api_key"))
    elif llm_vendor == "cohere":
        return CohereEmbeddings(cohere_api_key=get_secret("cohere_api_key"))


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
