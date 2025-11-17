#!/usr/bin/env python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from utils.searcher import SemanticSearcher

print("Loading searcher...")
searcher = SemanticSearcher()
print("Searcher loaded!")

print("Searching for 'vector embeddings'...")
results = searcher.search("vector embeddings", k=2)
print(f"Found {len(results)} results")

for result in results:
    print(f"Rank {result['rank']}: Score {result['score']:.4f}")
    print(f"  Source: {result.get('source', 'Unknown')}")
    print(f"  Text: {result.get('text', '')[:80]}...\n")
