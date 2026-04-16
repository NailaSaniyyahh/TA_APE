from flask import Flask, request, jsonify
from flask_cors import CORS
from opensearchpy import OpenSearch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
CORS(app)

# ─── Koneksi OpenSearch ───────────────────────────────────────────
client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_auth=("admin", "KoTA404TABAH!"),
    use_ssl=True,
    verify_certs=False,
    ssl_show_warn=False,
    timeout=60
)

INDEX_NAME = "books"

# ─── Load model SBERT sekali saja saat startup ───────────────────
print("Loading SBERT model...")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
print("SBERT model ready.")


# ════════════════════════════════════════════════════════════════════
# STEP 1 — Completion Suggester (OpenSearch)
# Tugasnya: dari query user, ambil kandidat judul buku
# ════════════════════════════════════════════════════════════════════
def get_candidates(query: str, size: int = 10) -> list[dict]:
    """
    Kirim query ke OpenSearch Completion Suggester.
    Return: list of dict {title, author, description}
    """
    body = {
        "suggest": {
            "suggest_by_title": {
                "prefix": query,
                "completion": {
                    "field": "suggest_title",
                    "size": size,
                    "skip_duplicates": True
                }
            },
            "suggest_by_author": {
                "prefix": query,
                "completion": {
                    "field": "suggest_author",
                    "size": size,
                    "skip_duplicates": True
                }
            }
        },
        "_source": ["title", "author", "description"]
    }

    response = client.search(index=INDEX_NAME, body=body)
    seen      = set()
    candidates = []
    # Prioritaskan hasil dari suggest_title dulu
    for suggest_key in ["suggest_by_title", "suggest_by_author"]:
        options = response["suggest"][suggest_key][0]["options"]
        for opt in options:
            src   = opt["_source"]
            title = src.get("title", "")
            if title in seen:
                continue
            seen.add(title)
            candidates.append({
                "title":       title,
                "author":      src.get("author", ""),
                "description": src.get("description", "")
            })

    return candidates[:size]


# ════════════════════════════════════════════════════════════════════
# STEP 2A — Re-ranking dengan BM25
# Tugasnya: hitung skor BM25 antara query vs tiap kandidat
# ════════════════════════════════════════════════════════════════════
# def rerank_bm25(query: str, candidates: list[dict]) -> list[dict]:
#     """
#     Tokenisasi setiap kandidat (title+author+description),
#     lalu hitung BM25 score terhadap query.
#     """
#     # Gabungkan field jadi satu dokumen teks per kandidat
#     corpus = [
#         f"{c['title']} {c['author']} {c['description']}"
#         for c in candidates
#     ]

#     # Tokenisasi sederhana (lowercase + split)
#     tokenized_corpus = [doc.lower().split() for doc in corpus]
#     tokenized_query  = query.lower().split()

#     bm25   = BM25Okapi(tokenized_corpus)
#     scores = bm25.get_scores(tokenized_query)

#     # Gabungkan skor ke kandidat
#     for i, c in enumerate(candidates):
#         c["score_bm25"] = float(scores[i])
#         c["score"]      = float(scores[i])  # score final

#     # Urutkan descending
#     return sorted(candidates, key=lambda x: x["score"], reverse=True)


# # ════════════════════════════════════════════════════════════════════
# # STEP 2B — Re-ranking dengan SBERT
# # Tugasnya: hitung cosine similarity antara embedding query vs kandidat
# # ════════════════════════════════════════════════════════════════════
# def rerank_sbert(query: str, candidates: list[dict]) -> list[dict]:
#     """
#     Encode query dan tiap kandidat pakai SBERT,
#     lalu hitung cosine similarity.
#     """
#     corpus = [
#         f"{c['title']} {c['author']} {c['description']}"
#         for c in candidates
#     ]

#     query_vec     = sbert_model.encode([query])
#     candidate_vec = sbert_model.encode(corpus)

#     similarities = cosine_similarity(query_vec, candidate_vec)[0]

#     for i, c in enumerate(candidates):
#         c["score_sbert"] = float(similarities[i])
#         c["score"]       = float(similarities[i])

#     return sorted(candidates, key=lambda x: x["score"], reverse=True)


# # ════════════════════════════════════════════════════════════════════
# # STEP 2C — Re-ranking Hybrid (BM25 + SBERT)
# # Tugasnya: normalisasi kedua skor lalu gabungkan dengan bobot
# # ════════════════════════════════════════════════════════════════════
# def rerank_hybrid(query: str, candidates: list[dict],
#                   alpha: float = 0.5) -> list[dict]:
#     """
#     alpha = bobot BM25 (0.5 berarti 50:50)
#     1 - alpha = bobot SBERT
#     """
#     # Hitung BM25
#     corpus = [
#         f"{c['title']} {c['author']} {c['description']}"
#         for c in candidates
#     ]
#     tokenized_corpus = [doc.lower().split() for doc in corpus]
#     tokenized_query  = query.lower().split()
#     bm25             = BM25Okapi(tokenized_corpus)
#     bm25_scores      = bm25.get_scores(tokenized_query)

#     # Hitung SBERT
#     query_vec     = sbert_model.encode([query])
#     candidate_vec = sbert_model.encode(corpus)
#     sbert_scores  = cosine_similarity(query_vec, candidate_vec)[0]

#     # Normalisasi BM25 ke rentang [0, 1]
#     bm25_max = max(bm25_scores) if max(bm25_scores) > 0 else 1
#     bm25_norm = bm25_scores / bm25_max

#     # SBERT sudah di rentang [-1, 1], normalisasi ke [0, 1]
#     sbert_norm = (sbert_scores + 1) / 2

#     # Gabungkan skor
#     for i, c in enumerate(candidates):
#         c["score_bm25"]  = float(bm25_scores[i])
#         c["score_sbert"] = float(sbert_scores[i])
#         c["score"]       = float(alpha * bm25_norm[i] +
#                                  (1 - alpha) * sbert_norm[i])

#     return sorted(candidates, key=lambda x: x["score"], reverse=True)


# ════════════════════════════════════════════════════════════════════
# ROUTES Flask
# ════════════════════════════════════════════════════════════════════

@app.route("/suggest", methods=["GET"])
def suggest():
    """
    GET /suggest?query=hunger&method=bm25
    Return: list kandidat yang sudah di-rerank
    """
    query  = request.args.get("query", "").strip()
    method = request.args.get("method", "bm25").lower()

    if not query:
        return jsonify([])

    # Step 1: ambil kandidat dari OpenSearch
    candidates = get_candidates(query, size=10)

    if not candidates:
        return jsonify([])

    # # Step 2: re-ranking sesuai metode
    # if method == "sbert":
    #     ranked = rerank_sbert(query, candidates)
    # elif method == "hybrid":
    #     ranked = rerank_hybrid(query, candidates)
    # else:
    #     ranked = rerank_bm25(query, candidates)

    return jsonify(candidates)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)