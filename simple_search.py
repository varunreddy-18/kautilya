# semantic_search.py
import os
import json
import argparse
import math

# Optional: suppress TF warnings (safe)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
except Exception:
    pass

from utils.embedder import embed_query
from utils.searcher import SemanticSearcher
from sentence_transformers import CrossEncoder

INDEX_PATH = "embeddings/index.faiss"
MAPPING_PATH = "embeddings/mappings.pkl"

def load_reranker(model_name="cross-encoder/ms-marco-MiniLM-L-12-v2"):
    try:
        reranker = CrossEncoder(model_name)
        return reranker
    except Exception as e:
        print("Warning: could not load reranker:", e)
        return None

def build_output(results, top_k=5, context_chars=800):
    out = []
    for r in results[:top_k]:
        meta = r["meta"]
        chunk_text = meta.get("chunk", "")
        preview = chunk_text if len(chunk_text) <= context_chars else chunk_text[:context_chars] + " ..."
        entry = {
            "doc_path": meta.get("doc_path"),
            "start_line": meta.get("start_line"),
            "end_line": meta.get("end_line"),
            "chunk_preview": preview,
            "index_score": r.get("index_score"),
            "rerank_score": r.get("rerank_score"),
            "rerank_prob": r.get("rerank_prob")
        }
        out.append(entry)
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--k", type=int, default=5, help="final top-k to return")
    parser.add_argument("--top_k_index", type=int, default=50, help="how many from index to fetch before rerank")
    parser.add_argument("--rerank_top_n", type=int, default=20, help="how many to rerank (<= top_k_index)")
    parser.add_argument("--debug_print_top", type=int, default=0, help="print top-N index results for debug (0=off)")
    args = parser.parse_args()

    if not os.path.exists(INDEX_PATH) or not os.path.exists(MAPPING_PATH):
        raise SystemExit("Index or mapping not found. Run build_index.py first.")

    searcher = SemanticSearcher(INDEX_PATH, MAPPING_PATH)

    q_vec = embed_query(args.query)  # normalized
    index_results = searcher.search(q_vec, top_k=args.top_k_index)

    # attach index_score for convenience
    for r in index_results:
        r["index_score"] = r["score"]

    # optional debug print of raw index hits
    if args.debug_print_top and len(index_results) > 0:
        print("\nTop index hits (raw):")
        for i, r in enumerate(index_results[:args.debug_print_top]):
            print(f"{i+1}. {r['meta'].get('doc_path')}  score={r['score']:.4f}")

    # Prepare texts for reranking
    reranker = load_reranker()
    candidates = index_results[:args.rerank_top_n]

    if reranker and len(candidates) > 0:
        pairs = [[args.query, c["meta"]["chunk"]] for c in candidates]
        try:
            rerank_scores = reranker.predict(pairs)
        except Exception as e:
            print("Reranker failed:", e)
            rerank_scores = None

        if rerank_scores is not None:
            # Convert logits to positive probabilities via softmax for stable comparison
            max_logit = float(max(rerank_scores))
            exp_scores = [math.exp(float(s) - max_logit) for s in rerank_scores]
            sum_exp = sum(exp_scores) if sum(exp_scores) > 0 else 1.0
            probs = [float(es / sum_exp) for es in exp_scores]

            for c, s, p in zip(candidates, rerank_scores, probs):
                c["rerank_score"] = float(s)   # raw logit
                c["rerank_prob"] = p
        else:
            # fallback to index score as prob-like
            for c in candidates:
                c["rerank_score"] = c["index_score"]
                c["rerank_prob"] = c["index_score"]
    else:
        # no reranker available -> use index_score min-max normalized
        all_scores = [c["score"] for c in index_results]
        if all_scores:
            mn, mx = min(all_scores), max(all_scores)
            rng = mx - mn if mx != mn else 1.0
        else:
            mn, mx, rng = 0.0, 1.0, 1.0
        for c in index_results[:args.rerank_top_n]:
            c["rerank_score"] = c["index_score"]
            c["rerank_prob"] = (c["index_score"] - mn) / rng
        candidates = index_results[:args.rerank_top_n]

    # sort by rerank_prob desc
    final_sorted = sorted(candidates, key=lambda x: x.get("rerank_prob", x.get("index_score", 0.0)), reverse=True)

    # --- PRINT TOP 5 SCORES CLEARLY ---
    print("\nTop 5 Scores:")
    for i, r in enumerate(final_sorted[:args.k]):
        idx_score = r.get("index_score")
        rerank_score = r.get("rerank_score")
        rerank_prob = r.get("rerank_prob")
        file_path = r["meta"].get("doc_path")
        print(
            f"{i+1}. index_score={idx_score:.6f}, "
            f"rerank_score={float(rerank_score):.6f}, "
            f"rerank_prob={float(rerank_prob):.6f}, "
            f"file={file_path}"
        )

    # JSON output (same as before)
    output = build_output(final_sorted, top_k=args.k)
    print(json.dumps({"query": args.query, "results": output}, indent=2))

if __name__ == "__main__":
    main()
