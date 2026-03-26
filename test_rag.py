import pytest
from query_data import query_rag
from langchain_ollama import OllamaLLM

EVAL_PROMPT = """You are a strict evaluator. Answer only 'true' or 'false'.

Expected: {expected_response}
Actual: {actual_response}

Does the actual response contain the key facts from the expected response? Reply with one word: true or false."""

LLM_MODEL = "llama3.2:3b"


def query_and_validate(question: str, expected_response: str) -> bool:
    response_text = query_rag(question)
    if response_text is None:
        print("\033[91mNo response returned.\033[0m")
        return False

    # Fast path: all key tokens present
    expected_tokens = expected_response.lower().replace("$", "").split()
    if all(tok in response_text.lower() for tok in expected_tokens):
        print("\033[92mPASS — exact match\033[0m")
        return True

    # Fallback: LLM-as-judge
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response,
        actual_response=response_text,
    )
    model = OllamaLLM(model=LLM_MODEL)
    result = model.invoke(prompt).strip().lower()

    if "true" in result:
        print(f"\033[92mPASS — llm eval: {result}\033[0m")
        return True
    elif "false" in result:
        print(f"\033[91mFAIL — llm eval: {result}\033[0m")
        return False
    else:
        raise ValueError(f"Unexpected eval result: '{result}'")


def test_monopoly_starting_money():
    assert query_and_validate(
        question="How much total money does a player start with in Monopoly?",
        expected_response="$1500",
    )


def test_ticket_to_ride_longest_train():
    assert query_and_validate(
        question="How many bonus points does the player with the Longest Continuous Path get in Ticket to Ride?",
        expected_response="10 points",
    )


def test_refuses_out_of_scope():
    """Should refuse to answer questions with no relevant context."""
    response = query_rag("What is the capital of Mars?")
    assert response is not None
    assert "don't have enough information" in response.lower()
