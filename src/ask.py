from config import agent_name, title_prompt
from model import load_store, setup_chain

print_cost: bool = True


def ask() -> None:
    store = load_store()
    chain = setup_chain(store, streaming=True)

    print(f"{agent_name}: {title_prompt}\n")

    streamed = False

    def write(token: str) -> None:
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
