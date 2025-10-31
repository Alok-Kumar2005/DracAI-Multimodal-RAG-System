def count_tokens(text: str) -> int:
    """Simple token counter based on word count."""
    if not text:
        return 0
    return len(text.split())


def count_tokens_batch(texts: list) -> list:
    """Count tokens for multiple texts."""
    return [count_tokens(text) for text in texts]