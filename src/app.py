"""Streamlit app for the LangChain demo."""

import streamlit as st
from langchain.schema.vectorstore import VectorStore

from .config import agent_name, title_prompt
from .model import load_store, setup_chain


def app() -> None:
    """Streamlit app for the LangChain demo."""
    st.set_page_config(page_title=agent_name, page_icon="🧠")

    st.title(agent_name)

    @st.cache_resource
    def loading() -> VectorStore:
        """Load the vector store."""
        return load_store()

    def add_message(role: str, content: str) -> None:
        """Add a message to the chat."""
        st.session_state.messages.append({"role": role, "content": content})

    if "chain" not in st.session_state:
        store = loading()
        chain = setup_chain(store, streaming=True)
        st.session_state.chain = chain

    if "messages" not in st.session_state:
        st.session_state.messages = []
        add_message("assistant", title_prompt)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input():
        add_message("user", question)

        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            answer = [""]

            def add_token(token: str) -> None:
                """Add a token to the answer."""
                answer[0] += token
                message_placeholder.markdown(answer[0])

            chain = st.session_state.chain
            answer[0] = chain(question, add_token)
            message_placeholder.markdown(answer[0])
        add_message("assistant", answer[0])
