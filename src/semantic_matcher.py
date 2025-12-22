# src/semantic_matcher.py

import numpy as np
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text: str):
    """
    Returns embedding for given text as numpy array
    """
    if not text:
        return np.zeros((1, _model.get_sentence_embedding_dimension()))
    return _model.encode([text], show_progress_bar=False)

def cosine_sim(vec1, vec2):
    """
    Compute cosine similarity between two vectors
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return 0.0

    return float(np.dot(v1, v2) / denom)
