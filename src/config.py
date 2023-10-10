# Where to store the Choma document embeddings
data_dir = "./data"

# The LLM vendor to use (openai, cohere)
llm_vendor = "cohere"

# The LLM model to use (gpt-3.5-turbo, command)
model_name = "command"

# Title tile/name shown in the chat window
agent_name = "Paul Graham"

# Skill description used for prompting
agent_skill = "You are an expert programmer, entrepreneur, and investor."

# The initial prompt by the agent
title_prompt = "How can I help you?"

# The LLM prompt used for agent response to user input
prompt_template = """
Use the following information to answer the question at the end.
If you don't know the answer, just say that you don't know without providing a reason. Don't try to make up an answer.
Use up to three paragraphs to answer the question. Be as concise as possible.
Provide one titled link to the most relevant source of your answer at the bottom if possible.

{context}

Question: {question}
Helpful Answer in markdown syntax:
""".strip()

document_template = "Title: {title}\nContent: {page_content}\nSource: {source}"
