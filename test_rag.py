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

    # Fast path: check if all key tokens from expected appear in response
    expected_tokens = expected_response.lower().replace("$", "").split()
    response_lower = response_text.lower()
    if all(tok in response_lower for tok in expected_tokens):
        print("\033[92m" + f"PASS — exact match" + "\033[0m")
        return True

    # Fallback: LLM-as-judge for fuzzy cases
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response,
        actual_response=response_text,
    )
    model = OllamaLLM(model=LLM_MODEL)
    result = model.invoke(prompt).strip().lower()

    if "true" in result:
        print("\033[92m" + f"PASS — llm eval: {result}" + "\033[0m")
        return True
    elif "false" in result:
        print("\033[91m" + f"FAIL — llm eval: {result}" + "\033[0m")
        return False
    else:
        raise ValueError(f"Unexpected eval result: '{result}'")


def test_monopoly_starting_money():
    assert query_and_validate(
        question="How much total money does a player start with in Monopoly? (Answer with the number only)",
        expected_response="$1500",
    )


def test_ticket_to_ride_longest_train():
    assert query_and_validate(
        question="How many bonus points does the player with the Longest Continuous Path get in Ticket to Ride?",
        expected_response="10 points",
    )
