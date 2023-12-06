from typing import Any

from evals.api import CompletionFn, CompletionResult

from src.model import load_store, setup_chain


class ChatCompletionResult(CompletionResult):
    def __init__(self, response) -> None:
        self.response = response

    def get_completions(self) -> list[str]:
        return [self.response]


class ChatCompletionFn(CompletionFn):
    def __init__(self, **kwargs) -> None:
        store = load_store()
        self.chain = setup_chain(store, streaming=False)

    def __call__(
        self,
        prompt: list[dict[str, str]],
        **kwargs,
    ) -> CompletionResult:
        question = prompt[0]["content"]
        answer = self.chain(question, None)
        return ChatCompletionResult(answer)
