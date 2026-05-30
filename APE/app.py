from flask import Flask, request, jsonify
from flask_cors import CORS
from opensearchpy import OpenSearch
from rank_bm25 import BM25Okapi
import copy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Setup logging untuk debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize OpenSearch with better error handling
try:
    client = OpenSearch(
        hosts=[{"host": "localhost", "port": 9200}],
        http_auth=("admin", "KoTA404TABAH!"),
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
        timeout=60,
        max_retries=3,
        retry_on_timeout=True
    )
    # Test connection
    client.info()
    logger.info("✓ OpenSearch connected successfully")
except Exception as e:
    logger.error(f"✗ Failed to connect to OpenSearch: {e}")
    logger.warning("App will continue but queries may fail")

INDEX_NAME = "books2"

sbert_model = None  

def get_sbert_model():
    global sbert_model
    if sbert_model is None:
        logger.info("📦 Loading SBERT model for the first time...")
        sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("✓ SBERT model loaded successfully")
    return sbert_model

def clear_sbert_memory():
    """Hapus model SBERT dari memory untuk menghemat resource"""
    global sbert_model
    if sbert_model is not None:
        import gc
        del sbert_model
        sbert_model = None
        gc.collect() 
        logger.info("🧹 SBERT model cleared from memory")

logger.info("⏳ Pre-loading SBERT ...")
get_sbert_model()
logger.info("✓ SBERT ready!")


def get_candidates(query: str) -> list[dict]:
    body = {
        "suggest": {
            "suggest_by_title": {
                "prefix": query,
                "completion": {
                    "field": "suggest_title",
                    "size": 10,
                    "skip_duplicates": True
                }
            },
            "suggest_by_author": {
                "prefix": query,
                "completion": {
                    "field": "suggest_author",
                    "size": 10,
                    "skip_duplicates": True
                }
            }
        },
        "query": {
            "bool": {
                "should":[
                    {
                        "match":{
                            "title":{
                                "query": query,
                                "fuzziness": "2"
                            }    
                        }
                    },
                    {
                        "match":{
                            "author":{
                                "query": query,
                                "fuzziness": "2"
                            }    
                        }
                    },    
                ],
                "minimum_should_match": "1"
            }
        },
        "size": 10,
        "_source": ["title", "author", "description", "numRatings", "pop_weight"]
    }

    response = client.search(index=INDEX_NAME, body=body)
    seen      = set()

    completion_result = []
    for key in ["suggest_by_title", "suggest_by_author"]:
        for opt in response["suggest"][key][0]["options"]:
            src   = opt["_source"]
            title = src.get("title", "")
            if not title or title in seen:
                continue
            seen.add(title)
            completion_result.append({
                "title":       title,
                "author":      src.get("author", ""),
                "description": src.get("description", ""),
                "numRatings":  int(src.get("numRatings", 0)),
                "pop_weight":  float(src.get("pop_weight", 0.0)),
                "source":      "completion"
            })
    match_result = []
    for hit in response.get("hits", {}).get("hits", []):
        src   = hit["_source"]
        title = src.get("title", "")
        if not title or title in seen:
            continue
        seen.add(title)
        match_result.append({
            "title":       title,
            "author":      src.get("author", ""),
            "description": src.get("description", ""),
            "numRatings":  int(src.get("numRatings", 0)),
            "pop_weight":  float(src.get("pop_weight", 0.0)),
            "source":      "fuzzy"
        })

    candidates = completion_result + match_result
    return candidates[:10]

def compute_bm25_scores(query, candidates):
    corpus = [f"{c['title']} {c['author']}" for c in candidates]
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    tokenized_query  = query.lower().split()
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25.get_scores(tokenized_query)  

def compute_sbert_scores(query, candidates):
    model = get_sbert_model()
    corpus = [f"{c['title']} {c['author']} {c['description']}" for c in candidates]
    query_vec     = model.encode([query])
    candidate_vec = model.encode(corpus)
    return cosine_similarity(query_vec, candidate_vec)[0]  # return scores saja

# def rerank_bm25(query: str, candidates: list[dict]) -> list[dict]:
#     if not candidates:
#         return candidates
    
#     # jadi doc
#     corpus = []
#     for c in candidates:
#         corpus.append(f"{c['title']} {c['author']}")

#     # tokenization
#     tokenized_corpus = []
#     for doc in corpus:
#         words = doc.lower().split()  # lowercase dulu, baru split
#         tokenized_corpus.append(words)

#     tokenized_query = query.lower().split()

#     bm25   = BM25Okapi(tokenized_corpus)   #bisa diganti jenisnya misal pake bm25L/bm25+
#     scores = bm25.get_scores(tokenized_query)

#     for i, candidate in enumerate(candidates):
#         candidate["score_bm25"] = float(scores[i])

#     reranked = sorted(candidates, key=lambda x: x["score_bm25"], reverse=True)

def normalize(scores):
    min_s, max_s = min(scores), max(scores)
    if max_s - min_s == 0:
        return [0.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]

#     return reranked
def rerank_bm25(query, candidates):
    if not candidates:         
        return candidates
    scores = compute_bm25_scores(query, candidates)
    for i, c in enumerate(candidates):
        c["score_bm25"] = float(scores[i])
    return sorted(candidates, key=lambda x: x["score_bm25"], reverse=True)

def rerank_sbert(query, candidates):
    if not candidates:         
        return candidates
    scores = compute_sbert_scores(query, candidates)
    for i, c in enumerate(candidates):
        c["score_sbert"] = float(scores[i])
    return sorted(candidates, key=lambda x: x["score_sbert"], reverse=True)

# def rerank_sbert(query: str, candidates: list[dict]) -> list[dict]:
#     if not candidates:
#         return candidates
    
#     # Encode query
#     query_embedding = sbert_model.encode(query, convert_to_tensor=False)
    
#     # Prepare documents: combine title, author, and description
#     documents = []
#     for c in candidates:
#         doc_text = f"{c['title']} {c['author']} {c['description']}"
#         documents.append(doc_text)
    
#     # Encode all documents
#     doc_embeddings = sbert_model.encode(documents, convert_to_tensor=False)
    
#     # Calculate cosine similarity between query and each document
#     scores = cosine_similarity([query_embedding], doc_embeddings)[0]
    
#     # Assign scores to candidates
#     for i, candidate in enumerate(candidates):
#         candidate["score_sbert"] = float(scores[i])
    
#     # Sort by SBERT score (descending)
#     reranked = sorted(candidates, key=lambda x: x["score_sbert"], reverse=True)
    
#     return reranked
# def rerank_sbert(query: str, candidates: list[dict]) -> list[dict]:
#     if not candidates:
#         return candidates
    
#     model = get_sbert_model()  # Load model jika belum ada
#     corpus = [f"{c['title']} {c['author']} {c['description']}" for c in candidates]
#     query_vec = model.encode([query])
#     candidate_vec = model.encode(corpus)
#     similarities = cosine_similarity(query_vec, candidate_vec)[0]
    
#     for i, c in enumerate(candidates):
#         c["score_sbert"] = float(similarities[i])
    
#     return sorted(candidates, key=lambda x: x["score_sbert"], reverse=True)

# def normalize_theoretical(scores, theoretical_min=0):
#     # """
#     # Theoretical min-max normalization.
#     # Formula: (score - theoretical_min) / (actual_max - theoretical_min)
    
#     # Args:
#     #     scores: list/array of scores to normalize
#     #     theoretical_min: theoretical minimum value (0 for BM25, -1 for SBERT cosine similarity)
    
#     # Returns:
#     #     list of normalized scores between [0, 1]
#     # """
#     if len(scores) == 0:
#         return []
    
#     actual_max = max(scores)
#     denominator = actual_max - theoretical_min
    
#     # Jika denominator 0 (semua skor sama), return array of 0s
#     if denominator == 0:
#         return [0.0] * len(scores)
    
#     return [(s - theoretical_min) / denominator for s in scores]

def rerank_hybrid(query, candidates, alpha=0.5, pop_boost=0.0):
    # alphanya 0.5 dulu buat nampilin si hasil hybridnya / default
    # if not candidates:
    #     return candidates
    
    # hitung skor bm25
    # corpus_bm25 = [f"{c['title']} {c['author']}" for c in candidates]
    # tokenized_corpus = [doc.lower().split() for doc in corpus_bm25]
    # tokenized_query  = query.lower().split()
    # bm25   = BM25Okapi(tokenized_corpus)
    # bm25_scores = bm25.get_scores(tokenized_query) 

    # # hitung skor sbert
    # model = get_sbert_model()  # Load model jika belum ada
    # corpus_sbert = [f"{c['title']} {c['author']} {c['description']}" for c in candidates]
    # query_vec     = model.encode([query])
    # candidate_vec = model.encode(corpus_sbert)
    # sbert_scores  = cosine_similarity(query_vec, candidate_vec)[0]
    # bm25_ranked   = rerank_bm25(query, copy.deepcopy(candidates))
    # sbert_ranked  = rerank_sbert(query, copy.deepcopy(candidates))
    
    # # Ekstrak scores dari hasil reranking
    # bm25_scores  = [c["score_bm25"] for c in bm25_ranked]
    # sbert_scores = [c["score_sbert"] for c in sbert_ranked]
    
    # print(f"\n{'='*60}")
    # print(f"DEBUG HYBRID — query: '{query}'")
    # print(f"{'='*60}")
    # print(f"{'Title':<35} {'BM25':>8} {'SBERT':>8}")
    # print(f"{'-'*55}")
    # for i, c in enumerate(candidates):
    #     print(f"{c['title'][:34]:<35} {bm25_scores[i]:>8.4f} {sbert_scores[i]:>8.4f}")
    # print(f"{'='*60}\n")

    # for i, c in enumerate(candidates):
    #     print(f"{c['title'][:34]:<35} {bm25_scores[i]:>8.4f} {sbert_scores[i]:>8.4f} {bm25_norm[i]:>8.4f} {sbert_norm[i]:>8.4f}")
    #     print(f"{'='*70}\n")
    # bm25_norm  = normalize_theoretical(bm25_scores, theoretical_min=0)
    # sbert_norm = normalize_theoretical(sbert_scores, theoretical_min=-1)

    # bm25_norm  = normalize(bm25_scores)
    # sbert_norm = normalize(sbert_scores)
    # # combine
    # for i, c in enumerate(candidates):
    #     c["score_bm25"]   = float(bm25_scores[i])
    #     c["score_sbert"]  = float(sbert_scores[i])
    #     c["score_hybrid"] = float((1 - alpha) * bm25_norm[i] + alpha * sbert_norm[i])
    
    # return sorted(candidates, key=lambda x: x["score_hybrid"], reverse=True)

    if not candidates:
        return candidates

    bm25_scores  = compute_bm25_scores(query, candidates)
    sbert_scores = compute_sbert_scores(query, candidates)

    print(f"\n{'='*65}")
    print(f"DEBUG HYBRID — query: '{query}', alpha={alpha}")
    print(f"{'Title':<35} {'BM25':>8} {'SBERT':>8}")
    print(f"{'-'*65}")
    for i, c in enumerate(candidates):
        print(f"{c['title'][:34]:<35} {float(bm25_scores[i]):>8.4f} {float(sbert_scores[i]):>8.4f}")
    print(f"{'='*65}\n")

    bm25_norm  = normalize(list(bm25_scores))
    sbert_norm = normalize([float(s) for s in sbert_scores])

    print(f"{'Title':<35} {'BM25N':>8} {'SBERTN':>8} {'HYBRID':>8}")
    print(f"{'-'*65}")
    for i, c in enumerate(candidates):
        hybrid = float((1 - alpha) * bm25_norm[i] + alpha * sbert_norm[i])
        print(f"{c['title'][:34]:<35} {bm25_norm[i]:>8.4f} {sbert_norm[i]:>8.4f} {hybrid:>8.4f}")
    print(f"{'='*65}\n")

    pop_scores = [c.get("pop_weight", 0.0) for c in candidates]
    pop_norm   = normalize(pop_scores) if max(pop_scores) > 0 else [0.0] * len(pop_scores)

    for i, c in enumerate(candidates):
        c["score_bm25"]   = float(bm25_scores[i])
        c["score_sbert"]  = float(sbert_scores[i])
        c["score_pop"]    = float(pop_scores[i])
        base_hybrid       = float((1 - alpha) * bm25_norm[i] + alpha * sbert_norm[i])
        c["score_hybrid"] = float(base_hybrid * (1 - pop_boost) + pop_norm[i] * pop_boost)

    return sorted(candidates, key=lambda x: x["score_hybrid"], reverse=True)

@app.route("/suggest", methods=["GET"])
def suggest():
    """
    GET /suggest?query=hunger&method=bm25
    GET /suggest?query=hunger&method=sbert
    Return: list kandidat yang sudah di-rerank
    
    Methods:
    - bm25: BM25 lexical matching
    - sbert: Semantic search using SBERT (all-MiniLM-L6-v2)
    """
    try:
        query  = request.args.get("query", "").strip()
        method = request.args.get("method", "bm25").lower()

        if not query:
            return jsonify([])

        candidates = get_candidates(query)

        if not candidates:
            return jsonify([])

        if method == "sbert":
            ranked = rerank_sbert(query, candidates)
        else:  # default to bm25
            ranked = rerank_bm25(query, candidates)

        return jsonify(ranked)
    except Exception as e:
        logger.error(f"Error in /suggest: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/compare_completion_bm25", methods=["GET"])
def compare_completion_bm25():
    """
    Compare completion and BM25 ranking methods
    GET /compare_completion_bm25?query=hunger
    """
    query = request.args.get("query", "").strip()
    if not query:
        return jsonify({"completion": [], "bm25": []})

    candidates = get_candidates(query)
    if not candidates:
        return jsonify({"completion": [], "bm25": []})
    completion_order = copy.deepcopy(candidates)

    # Re-ranking BM25
    bm25_order = rerank_bm25(query, copy.deepcopy(candidates))

    return jsonify({
        "completion": completion_order,
        "bm25": bm25_order
    })


@app.route("/compare_all_methods", methods=["GET"])
def compare_all_methods():
    query     = request.args.get("query", "").strip()
    alpha     = float(request.args.get("alpha", 0.5))
    pop_boost = float(request.args.get("pop_boost", 0.0))

    if not query:
        return jsonify({"completion": [], "bm25": [], "sbert": [], "hybrid": []})

    candidates = get_candidates(query)
    if not candidates:
        return jsonify({"completion": [], "bm25": [], "sbert": [], "hybrid": []})

    completion_order = copy.deepcopy(candidates)
    bm25_order       = rerank_bm25(query,   copy.deepcopy(candidates))
    sbert_order      = rerank_sbert(query,  copy.deepcopy(candidates))
    hybrid_order     = rerank_hybrid(query, copy.deepcopy(candidates), alpha=alpha, pop_boost=pop_boost)

    # Hapus description dari response
    for lst in [completion_order, bm25_order, sbert_order, hybrid_order]:
        for c in lst:
            c.pop("description", None)

    return jsonify({
        "completion": completion_order,
        "bm25":       bm25_order,
        "sbert":      sbert_order,
        "hybrid":     hybrid_order
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/sbert-status", methods=["GET"])
def sbert_status():
    global sbert_model
    is_loaded = sbert_model is not None
    return jsonify({
        "model_loaded": is_loaded,
        "message": "✓ SBERT model is in memory" if is_loaded else "✗ SBERT model not loaded"
    })


@app.route("/init-sbert", methods=["POST"])
def init_sbert():
    """Warmup endpoint (dipanggil dari HTML saat page load)"""
    try:
        get_sbert_model()  # sudah ter-load saat startup, ini hanya konfirmasi
        return jsonify({
            "message": "✓ SBERT model ready!",
            "status": "ready",
            "model_loaded": True
        })
    except Exception as e:
        logger.error(f"Error initializing SBERT: {e}")
        return jsonify({"error": str(e), "status": "failed"}), 500


@app.route("/clear-sbert", methods=["POST"])
def clear_sbert():
    clear_sbert_memory()
    return jsonify({
        "message": "✓ SBERT model cleared from memory",
        "note": "Model akan di-load ulang saat dibutuhkan"
    })
    


if __name__ == "__main__":
    app.run(debug=False, threaded=True, port=5000, use_reloader=False)