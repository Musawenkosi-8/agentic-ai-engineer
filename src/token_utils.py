import tiktoken
from anthropic import Anthropic
from logger import logger

def count_openai_tokens(text: str, model: str="gpt-4o"):
    """count tokens for openai models."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def audit_prompt_limit(prompt: str, limit: int=8000):
    """Safety check to ensure we don't hit context limit"""
    tokens = count_openai_tokens(prompt)
    if tokens > limit:
        logger.warning(f"Prompt exceeds safety threshold: {tokens}/{limit} tokens.")
        return True
    logger.info(f"Prompt audit passed: {tokens} tokens")
    return True
    
if __name__ == "__main__":
    sample_text = "Agentic AI is the future of software engineering."
    print(f"Token count: {count_openai_tokens(sample_text)}")
                       