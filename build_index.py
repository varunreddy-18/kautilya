# build_index.py
import os
import pickle
from tqdm import tqdm
import numpy as np
import faiss

from utils.loader import load_documents
from utils.chunker import split_on_headings_and_windows
from utils.embedder import embed_texts

DATA_DIR = "data/twitter_docs"
EMBED_DIR = "embeddings"
INDEX_PATH = os.path.join(EMBED_DIR, "index.faiss")
MAPPING_PATH = os.path.join(EMBED_DIR, "mappings.pkl")

os.makedirs(EMBED_DIR, exist_ok=True)

print("Loading documents...")
docs = load_documents(DATA_DIR)

all_chunks = []
mapping = {}
idx = 0

print("Chunking documents...")
for d in tqdm(docs):

    # NO FILTERS – DO NOT REMOVE ANY FILES
    chunks = split_on_headings_and_windows(d["lines"], max_words=350)

    for c in chunks:
        if len(c["chunk"].split()) < 10:
            continue

        mapping[idx] = {
            "doc_path": d["path"],
            "chunk": c["chunk"],
            "start_line": c["start_line"],
            "end_line": c["end_line"]
        }
        all_chunks.append(c["chunk"])
        idx += 1

print("Total chunks:", len(all_chunks))
if len(all_chunks) == 0:
    raise SystemExit("❌ No chunks created – your documentation folder is EMPTY!")

print("Embedding chunks...")
vectors = embed_texts(all_chunks)

dim = vectors.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(vectors)

faiss.write_index(index, INDEX_PATH)
with open(MAPPING_PATH, "wb") as fp:
    pickle.dump(mapping, fp)

print("✅ Index built successfully.")
print("Chunks:", len(all_chunks))
print("Index saved to:", INDEX_PATH)
