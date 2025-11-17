# utils/loader.py
import os

def load_documents(root_path, allowed_ext=(".md", ".markdown", ".txt", ".json")):
    """
    Walk root_path and return list of dicts: {"path": path, "lines": [line1, line2, ...], "text": full_text}
    """
    docs = []
    for root, _, files in os.walk(root_path):
        for fname in files:
            if not fname.lower().endswith(allowed_ext):
                continue
            path = os.path.join(root, fname)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                    text = "".join(lines)
            except Exception:
                continue
            docs.append({"path": path, "lines": lines, "text": text})
    return docs
