from langchain.callbacks import get_openai_callback
from config import title_prompt, agent_name
from util import load_store, setup_chain

print_cost = True


def ask():
    store = load_store()
    chain = setup_chain(store, streaming=False)

    print(f"{agent_name}: {title_prompt}\n")

    while True:
        question = input("You: ")
        if len(question) == 0:
            break
        with get_openai_callback() as callback:
            answer = chain(question)
            cost = callback.total_cost
        print(f"\n{agent_name}: {answer}\n")
        if print_cost:
            print(f"Cost: ${cost:.3f}\n")


if __name__ == "__main__":
    ask()
