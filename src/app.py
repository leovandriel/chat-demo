import streamlit as st
from config import agent_name, title_prompt
from util import setup_chain

st.set_page_config(page_title=agent_name, page_icon="🧠")

st.title(agent_name)


@st.cache_resource
def loading():
    return setup_chain()


with st.form(key='form'):
    question = st.text_input(title_prompt)
    submitted = st.form_submit_button('Ask')
    chain = loading()
    if submitted:
        answer = chain.invoke(question).content
        st.info(answer)
