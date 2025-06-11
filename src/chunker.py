import tiktoken


ENC = tiktoken.get_encoding("cl100k_base")

def encode_tokens(text: str, max_tokens: int = 2000):
    """
    Yield successive chunks of `text`, each no longer than `max_tokens`
    according to the tiktoken (cl100k_base) encoding.
    """
    tokens = ENC.encode(text)
    for i in range(0, len(tokens), max_tokens):
        yield ENC.decode(tokens[i : i + max_tokens])
