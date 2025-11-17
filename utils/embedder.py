# utils/embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np

# Choose a small, fast model for embeddings. Change if you prefer another.
_EMB_MODEL_NAME = "all-MiniLM-L6-v2"
_model = SentenceTransformer(_EMB_MODEL_NAME)

def embed_texts(texts, batch_size=64):
    """
    Returns numpy array of shape (n_texts, dim) dtype float32, normalized (L2)
    """
    vectors = _model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)
    vectors = vectors.astype("float32")
    # L2 normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    vectors = vectors / norms
    return vectors

def embed_query(text):
    v = _model.encode([text], convert_to_numpy=True)
    v = v.astype("float32")
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    norm[norm == 0] = 1e-12
    return v / norm
