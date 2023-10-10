from config import title_prompt, agent_name
from model import load_store, setup_chain

print_cost = True


def ask():
    store = load_store()
    chain = setup_chain(store, streaming=False)

    print(f"{agent_name}: {title_prompt}\n")

    while True:
        question = input("You: ")
        if len(question) == 0:
            break
        answer = chain(question)
        print(f"\n{agent_name}: {answer}\n")


if __name__ == "__main__":
    ask()
