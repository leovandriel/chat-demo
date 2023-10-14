import streamlit as st
from langchain.schema.vectorstore import VectorStore
from config import agent_name, title_prompt
from model import load_store, setup_chain


st.set_page_config(page_title=agent_name, page_icon="🧠")

st.title(agent_name)


@st.cache_resource
def loading() -> VectorStore:
    return load_store()


def add_message(role: str, content: str) -> None:
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
        answer = ""

        def add_token(token: str) -> None:
            global answer
            answer += token
            message_placeholder.markdown(answer)

        chain = st.session_state.chain
        answer = chain(question, add_token)
        message_placeholder.markdown(answer)
    add_message("assistant", answer)
