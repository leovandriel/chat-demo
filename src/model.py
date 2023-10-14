"""Model setup and the chain setup."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable, cast

from langchain.callbacks.base import BaseCallbackHandler, Callbacks
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import (
    ChatAnthropic,
    ChatCohere,
    ChatGooglePalm,
    ChatOpenAI,
    ChatVertexAI,
)
from langchain.embeddings import (
    CohereEmbeddings,
    GooglePalmEmbeddings,
    OpenAIEmbeddings,
    VertexAIEmbeddings,
)
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema.vectorstore import VectorStore
from langchain.vectorstores import Chroma
from pydantic.v1 import SecretStr

from .config import agent_skill, data_dir, document_template, prompt_template
from .util import get_secret

if TYPE_CHECKING:
    from langchain.chat_models.base import BaseChatModel
    from langchain.embeddings.base import Embeddings


def create_llm(
    vendor: str,
    model: str,
    *,
    temperature: float = 0.5,
    streaming: bool = False,
    callbacks: Callbacks = None,
) -> BaseChatModel:
    """Create a language model."""
    if vendor == "openai":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=get_secret("openai_api_key"),
            streaming=streaming,
            callbacks=callbacks,
        )
    if vendor == "cohere":
        return ChatCohere(
            model=model,
            temperature=temperature,
            cohere_api_key=get_secret("cohere_api_key"),
            streaming=streaming,
            callbacks=callbacks,
            client=None,
            async_client=None,
        )
    if vendor == "anthropic":
        return ChatAnthropic(
            model_name=model,
            temperature=temperature,
            anthropic_api_key=SecretStr(get_secret("anthropic_api_key")),
            streaming=streaming,
            callbacks=callbacks,
        )
    if vendor == "google":
        return ChatGooglePalm(
            model_name=model,
            temperature=temperature,
            google_api_key=get_secret("google_api_key"),
            callbacks=callbacks,
            client=None,
        )
    if vendor == "vertex":
        from google.oauth2.service_account import (  # type: ignore[import-untyped]
            Credentials,
        )

        info = json.loads(get_secret("vertex_api_key"))
        return ChatVertexAI(
            model_name=model,
            temperature=temperature,
            credentials=Credentials.from_service_account_info(info),
            project=info["project_id"],
            streaming=streaming,
            callbacks=callbacks,
        )
    msg = f"Unknown vendor: {vendor}"
    raise ValueError(msg)


def create_embedder(vendor: str, model: str) -> Embeddings:
    """Create an embeddings model."""
    if vendor == "openai":
        return OpenAIEmbeddings(
            model=model,
            openai_api_key=get_secret("openai_api_key"),
        )
    if vendor == "cohere":
        return CohereEmbeddings(
            model=model,
            cohere_api_key=get_secret("cohere_api_key"),
            client=None,
            async_client=None,
        )
    if vendor == "google":
        return GooglePalmEmbeddings(
            model_name=model,
            google_api_key=get_secret("google_api_key"),
            client=None,
        )
    if vendor == "vertex":
        from google.oauth2.service_account import Credentials

        info = json.loads(get_secret("vertex_api_key"))
        return VertexAIEmbeddings(
            model_name=model,
            credentials=Credentials.from_service_account_info(info),
            project=info["project_id"],
        )
    msg = f"Unknown vendor: {vendor}"
    raise ValueError(msg)


def load_store() -> VectorStore:
    """Load the vector store."""
    embedder = create_embedder(vendor="openai", model="text-embedding-ada-002")
    vectorstore = Chroma(
        embedding_function=embedder,
        persist_directory=data_dir.as_posix(),
    )
    return cast(VectorStore, vectorstore)


class StreamHandler(BaseCallbackHandler):
    """Handler for the stream."""

    callback: Callable[[str], None] | None

    def __init__(self: StreamHandler) -> None:
        """Initialize the handler."""
        self.callback = None

    def on_llm_new_token(
        self: StreamHandler,
        token: str,
        **_: Any,  # noqa: ANN401
    ) -> None:
        """Handle a new token."""
        if self.callback is not None:
            self.callback(token)

    def on_chat_model_start(
        self: StreamHandler,
        *args: Any,  # noqa: ANN401
        **_: Any,  # noqa: ANN401
    ) -> None:
        """Handle the start of the chat model."""


def setup_chain(
    vectorstore: VectorStore,
    *,
    streaming: bool,
) -> Callable[[str, Callable[[str], None]], str]:
    """Set up the chain."""
    retriever = vectorstore.as_retriever()

    internal_llm = create_llm(vendor="openai", model="gpt-3.5-turbo", temperature=0.2)

    memory = ConversationSummaryBufferMemory(
        llm=internal_llm,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    prompt = PromptTemplate.from_template(f"{agent_skill}\n{prompt_template}")
    document_prompt = PromptTemplate.from_template(document_template)
    handler = StreamHandler()

    response_llm = create_llm(
        vendor="openai",
        model="gpt-3.5-turbo",
        temperature=0.5,
        streaming=streaming,
        callbacks=[handler],
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

    def run(question: str, callback: Callable[[str], None] | None = None) -> str:
        handler.callback = callback
        return cast(str, chain({"question": question})["answer"])

    return run
