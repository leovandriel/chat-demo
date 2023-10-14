"""CLI version of the chat app."""

import sys

from .config import agent_name, title_prompt
from .model import load_store, setup_chain

print_cost: bool = True


def ask() -> None:
    """Ask the agent questions and get answers."""
    store = load_store()
    chain = setup_chain(store, streaming=True)

    sys.stdout.write(f"{agent_name}: {title_prompt}\n\n")

    streamed = False

    def write(token: str) -> None:
        nonlocal streamed
        sys.stdout.write(token)
        sys.stdout.flush()
        streamed = True

    while True:
        question = input("You: ")
        if len(question) == 0:
            break
        sys.stdout.write(f"\n{agent_name}: ")
        answer = chain(question, write)
        if not streamed:
            sys.stdout.write(f"{answer}\n")
        sys.stdout.write("\n\n")
        sys.stdout.flush()
