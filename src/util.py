import os
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from config import data_dir, model_name, agent_skill, prompt_template, document_template


def get_secret(key):
    filename = f"{key}.txt"
    if os.path.isfile(filename):
        with open(filename) as f:
            return f.read().strip()
    else:
        raise Exception(
            f"Unable to find {key}.txt. Please create this file and add your secret."
        )


class StreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.callback = None

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.callback(token)


def load_store():
    embedder = OpenAIEmbeddings(openai_api_key=get_secret("openai_api_key"))
    vectorstore = Chroma(
        embedding_function=embedder,
        persist_directory=data_dir,
    )
    return vectorstore


def setup_chain(vectorstore, streaming):
    retriever = vectorstore.as_retriever()

    internal_llm = ChatOpenAI(
        model_name=model_name,
        temperature=0,
        openai_api_key=get_secret("openai_api_key"),
    )

    memory = ConversationSummaryBufferMemory(
        llm=internal_llm,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    prompt = PromptTemplate.from_template("\n".join([agent_skill, prompt_template]))
    document_prompt = PromptTemplate.from_template(document_template)
    handler = StreamHandler()

    response_llm = ChatOpenAI(
        model_name=model_name,
        temperature=0,
        openai_api_key=get_secret("openai_api_key"),
        streaming=streaming,
        callbacks=[handler] if handler else [],
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
