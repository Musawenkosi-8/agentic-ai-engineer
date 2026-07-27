import tiktoken
from src.logger import logger


def count_tokens(text: str, model: str = "gpt-4o"):
    """
    Estimate the number of tokens in a piece of text.

    Although this project uses Groq, tiktoken provides a good
    approximation for token counting.
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def audit_prompt_limit(prompt: str, limit: int = 8000):
    """
    Check whether a prompt exceeds the desired token limit.
    """
    tokens = count_tokens(prompt)

    if tokens > limit:
        logger.warning(
            f"Prompt exceeds safety threshold: {tokens}/{limit} tokens."
        )
        return False

    logger.info(f"Prompt audit passed: {tokens}/{limit} tokens.")
    return True


if __name__ == "__main__":
    sample_text = "Agentic AI is the future of software engineering."

    token_count = count_tokens(sample_text)

    print(f"Token count: {token_count}")

    audit_prompt_limit(sample_text)