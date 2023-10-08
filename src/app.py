import streamlit as st
from ask import load_store, setup_chain, agent_name, title_prompt

st.set_page_config(page_title=agent_name, page_icon="🧠")

st.title(agent_name)


@st.cache_resource
def loading():
    return load_store()


if 'chain' not in st.session_state:
    store = loading()
    chain = setup_chain(store, streaming=True)
    st.session_state['chain'] = chain

if 'messages' not in st.session_state:
    st.session_state.messages = [
        {'role': 'assistant', 'content': title_prompt}]

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if question := st.chat_input():
    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message('user'):
        st.markdown(question)
    with st.chat_message('assistant'):
        message_placeholder = st.empty()
        answer = ''

        def add_token(token):
            global answer
            answer += token
            message_placeholder.markdown(answer)
        chain = st.session_state['chain']
        answer = chain(question, add_token)
        message_placeholder.markdown(answer)
    st.session_state.messages.append(
        {'role': 'assistant', 'content': answer})
