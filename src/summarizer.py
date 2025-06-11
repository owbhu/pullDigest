from transformers import pipeline
from src.config import HF_MODEL
import torch

device = 0 if torch.backends.mps.is_available() else -1

summarizer = pipeline(
    "summarization",
    model=HF_MODEL,
    device=device
)

def summarize_chunk(chunk: str, max_len: int = 75, min_len: int = 15) -> str:
    """
    Summarize a chunk, using tighter length bounds to avoid warnings.
    """
    out = summarizer(
        chunk,
        max_length=max_len,
        min_length=min_len,
        do_sample=False,
        truncation=True
    )
    return out[0]["summary_text"]
