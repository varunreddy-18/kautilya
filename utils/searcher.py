# utils/searcher.py
import faiss
import pickle
import numpy as np

class SemanticSearcher:
    def __init__(self, index_path, mapping_path):
        self.index = faiss.read_index(index_path)
        with open(mapping_path, "rb") as fp:
            self.mapping = pickle.load(fp)   # mapping: idx -> metadata dict

    def search(self, q_vec, top_k=50):
        """
        q_vec: numpy array shape (1, dim) float32, normalized
        Returns list of dicts with keys: idx, score (inner product), metadata
        """
        D, I = self.index.search(q_vec, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            meta = self.mapping.get(int(idx), {})
            results.append({
                "idx": int(idx),
                "score": float(score),
                "meta": meta
            })
        return results
