import os
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import Chroma
from config import data_dir, agent_skill, prompt_template, model_name

openai_key_file = 'openai_api_key.txt'


def openai_key():
    if os.path.isfile(openai_key_file):
        with open(openai_key_file) as f:
            return f.read().strip()
    else:
        raise Exception(
            f'Please create a file called {openai_key_file} with your OpenAI API key in it.')


def setup_chain():
    embedder = OpenAIEmbeddings(openai_api_key=openai_key())
    vectorstore = Chroma(
        embedding_function=embedder,
        persist_directory=data_dir,
    )

    llm = ChatOpenAI(
        model_name=model_name,
        temperature=0,
        openai_api_key=openai_key(),
    )

    retriever = vectorstore.as_retriever()
    prompt = PromptTemplate.from_template(
        "\n".join([agent_skill, prompt_template]))
    chain = (
        {'context': retriever, 'question': RunnablePassthrough()}
        | prompt
        | llm
    )

    return chain
