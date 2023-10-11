from config import title_prompt, agent_name
from model import load_store, setup_chain

print_cost = True


def ask():
    store = load_store()
    chain = setup_chain(store, streaming=True)

    print(f"{agent_name}: {title_prompt}\n")

    streamed = False

    def write(token):
        nonlocal streamed
        print(token, end="", flush=True)
        streamed = True

    while True:
        question = input("You: ")
        if len(question) == 0:
            break
        print(f"\n{agent_name}: ", end="", flush=True)
        answer = chain(question, write)
        if not streamed:
            print(answer, end="")
        print("\n")


if __name__ == "__main__":
    ask()
