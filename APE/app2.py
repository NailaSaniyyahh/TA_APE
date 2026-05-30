from flask import Flask, request, jsonify
from flask_cors import CORS
from opensearchpy import OpenSearch
from rank_bm25 import BM25Okapi
import copy, json, os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import pathlib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app  = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────────────────────────
# OPENSEARCH CONNECTION
# ─────────────────────────────────────────────────────────────────
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
    client.info()
    logger.info("✓ OpenSearch connected")
except Exception as e:
    logger.error(f"✗ OpenSearch failed: {e}")

INDEX_NAME  = "books2"
sbert_model = None

def get_sbert_model():
    global sbert_model
    if sbert_model is None:
        logger.info("📦 Loading SBERT model...")
        sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("✓ SBERT model loaded")
    return sbert_model

def clear_sbert_memory():
    global sbert_model
    if sbert_model is not None:
        import gc
        del sbert_model
        sbert_model = None
        gc.collect()
        logger.info("🧹 SBERT cleared from memory")

logger.info("⏳ Pre-loading SBERT...")
get_sbert_model()
logger.info("✓ SBERT ready!")


# ─────────────────────────────────────────────────────────────────
# CANDIDATE GENERATION
# Mengambil tepat 10 kandidat dari OpenSearch
# via completion suggester + fuzzy sebagai fallback
# ─────────────────────────────────────────────────────────────────
def get_candidates(query: str) -> list[dict]:
    """
    Menghasilkan tepat 10 kandidat unik dari OpenSearch.
    Prioritas: completion suggester (suggest_title + suggest_author)
    Fallback  : fuzzy match jika completion < 10
    """
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
                "should": [
                    {"match": {"title":  {"query": query, "fuzziness": "2"}}},
                    {"match": {"author": {"query": query, "fuzziness": "2"}}}
                ],
                "minimum_should_match": "1"
            }
        },
        "size": 10,
        "_source": ["title", "author", "description", "numRatings", "pop_weight"]
    }

    response = client.search(index=INDEX_NAME, body=body)
    seen     = set()
    result   = []

    # Completion suggester (prioritas utama)
    for key in ["suggest_by_title", "suggest_by_author"]:
        for opt in response["suggest"][key][0]["options"]:
            if len(result) >= 10:
                break
            src   = opt["_source"]
            title = src.get("title", "")
            if not title or title in seen:
                continue
            seen.add(title)
            result.append({
                "title":       title,
                "author":      src.get("author", ""),
                "description": src.get("description", ""),
                "numRatings":  int(src.get("numRatings", 0)),
                "pop_weight":  float(src.get("pop_weight", 0.0)),
                "source":      "completion"
            })

    # Fuzzy match fallback jika completion < 10
    if len(result) < 10:
        for hit in response.get("hits", {}).get("hits", []):
            if len(result) >= 10:
                break
            src   = hit["_source"]
            title = src.get("title", "")
            if not title or title in seen:
                continue
            seen.add(title)
            result.append({
                "title":       title,
                "author":      src.get("author", ""),
                "description": src.get("description", ""),
                "numRatings":  int(src.get("numRatings", 0)),
                "pop_weight":  float(src.get("pop_weight", 0.0)),
                "source":      "fuzzy"
            })

    return result


# ─────────────────────────────────────────────────────────────────
# SCORING & NORMALIZATION
# ─────────────────────────────────────────────────────────────────
def compute_bm25_scores(query, candidates):
    corpus           = [f"{c['title']} {c['author']}" for c in candidates]
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    tokenized_query  = query.lower().split()
    bm25             = BM25Okapi(tokenized_corpus)
    return bm25.get_scores(tokenized_query)

def compute_sbert_scores(query, candidates):
    model         = get_sbert_model()
    corpus        = [f"{c['title']} {c['author']} {c['description']}" for c in candidates]
    query_vec     = model.encode([query])
    candidate_vec = model.encode(corpus)
    return cosine_similarity(query_vec, candidate_vec)[0]

def normalize(scores):
    min_s, max_s = min(scores), max(scores)
    if max_s - min_s == 0:
        return [0.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


# ─────────────────────────────────────────────────────────────────
# RE-RANKING
# ─────────────────────────────────────────────────────────────────
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

def rerank_hybrid(query, candidates, alpha=0.7, pop_boost=0.0):
    if not candidates:
        return candidates

    bm25_scores  = compute_bm25_scores(query, candidates)
    sbert_scores = compute_sbert_scores(query, candidates)
    bm25_norm    = normalize(list(bm25_scores))
    sbert_norm   = normalize([float(s) for s in sbert_scores])

    pop_scores = [c.get("pop_weight", 0.0) for c in candidates]
    pop_norm   = normalize(pop_scores) if max(pop_scores) > 0 else [0.0] * len(pop_scores)

    print(f"\n{'='*65}")
    print(f"DEBUG HYBRID — query: '{query}', alpha={alpha}, pop_boost={pop_boost}")
    print(f"{'Title':<35} {'BM25':>8} {'SBERT':>8} {'BM25N':>8} {'SBERTN':>8} {'POPW':>8} {'HYBRID':>8}")
    print(f"{'-'*65}")
    for i, c in enumerate(candidates):
        base   = float((1 - alpha) * bm25_norm[i] + alpha * sbert_norm[i])
        hybrid = float(base * (1 - pop_boost) + pop_norm[i] * pop_boost)
        print(f"{c['title'][:34]:<35} {float(bm25_scores[i]):>8.4f} {float(sbert_scores[i]):>8.4f} {bm25_norm[i]:>8.4f} {sbert_norm[i]:>8.4f} {pop_norm[i]:>8.4f} {hybrid:>8.4f}")
    print(f"{'='*65}\n")

    for i, c in enumerate(candidates):
        base_hybrid       = float((1 - alpha) * bm25_norm[i] + alpha * sbert_norm[i])
        c["score_bm25"]   = float(bm25_scores[i])
        c["score_sbert"]  = float(sbert_scores[i])
        c["score_pop"]    = float(pop_scores[i])
        c["score_hybrid"] = float(base_hybrid * (1 - pop_boost) + pop_norm[i] * pop_boost)

    return sorted(candidates, key=lambda x: x["score_hybrid"], reverse=True)


# ─────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────

@app.route("/compare_all_methods", methods=["GET"])
def compare_all_methods():
    """
    GET /compare_all_methods?query=hunger&alpha=0.7

    Dipakai oleh:
    - index.html : visualisasi 4 kolom (alpha default 0.7)
    - /run_experiment : otomasi eksperimen batch
    """
    try:
        query     = request.args.get("query", "").strip()
        alpha     = float(request.args.get("alpha", 0.7))
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

        # Hapus description dari response (tidak perlu di frontend)
        for lst in [completion_order, bm25_order, sbert_order, hybrid_order]:
            for c in lst:
                c.pop("description", None)

        return jsonify({
            "completion": completion_order,
            "bm25":       bm25_order,
            "sbert":      sbert_order,
            "hybrid":     hybrid_order
        })

    except Exception as e:
        logger.error(f"Error /compare_all_methods: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/run_experiment", methods=["GET"])
def run_experiment():
    """
    GET /run_experiment?alpha=0.60

    Endpoint otomasi eksperimen — menggantikan candidate_generation.py.
    Membaca test_queries_final.json, jalankan semua query dengan
    alpha yang diberikan, simpan hasilnya ke JSON.

    Cara pakai:
      http://localhost:5000/run_experiment?alpha=0.60
      http://localhost:5000/run_experiment?alpha=0.65
      http://localhost:5000/run_experiment?alpha=0.70
      http://localhost:5000/run_experiment?alpha=0.75
      http://localhost:5000/run_experiment?alpha=0.80
    """
    try:
        alpha      = float(request.args.get("alpha", 0.7))
        BASE_DIR   = Path(__file__).resolve().parent
        input_file = BASE_DIR / "../../DATASET/v3test_queries_final.json"
        # input_file = "../../test_queries_final.json"

        if not os.path.exists(input_file):
            return jsonify({"error": f"File {input_file} tidak ditemukan"}), 404

        with open(input_file, "r", encoding="utf-8") as f:
            queries = json.load(f)

        alpha_str   = f"{alpha:.2f}".replace(".", "")
        output_file = f"v4experiment_results_alpha{alpha_str}.json"

        results = []
        for i, q in enumerate(queries, 1):
            qtext = q["query_text"]
            logger.info(f"[{i:3}/{len(queries)}] Processing: '{qtext}'")

            candidates = get_candidates(qtext)

            if not candidates:
                results.append({
                    "query_id"   : q["query_id"],
                    "query_text" : qtext,
                    "query_type" : q["query_type"],
                    "title"      : q.get("title", ""),
                    "author"     : q.get("author", ""),
                    "alpha"      : alpha,
                    "status"     : "FAILED",
                    "candidates" : {
                        "completion": []
                    }
                })
                continue

            completion_order = copy.deepcopy(candidates)
            # bm25_order       = rerank_bm25(qtext,   copy.deepcopy(candidates))
            # sbert_order      = rerank_sbert(qtext,  copy.deepcopy(candidates))
            # hybrid_order     = rerank_hybrid(qtext, copy.deepcopy(candidates), alpha=alpha)

            # Hapus description dari hasil
            for lst in [completion_order]:
                for c in lst:
                    c.pop("description", None)

            results.append({
                "query_id"   : q["query_id"],
                "query_text" : qtext,
                "query_type" : q["query_type"],
                "title"      : q.get("title", ""),
                "author"     : q.get("author", ""),
                "alpha"      : alpha,
                "status"     : "OK",
                "candidates" : {
                    "completion": completion_order
                }
            })

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        success = sum(1 for r in results if r["status"] == "OK")
        failed  = len(results) - success

        logger.info(f"✓ Eksperimen selesai → {output_file}")
        return jsonify({
            "message"    : f"Eksperimen selesai dengan alpha={alpha}",
            "alpha"      : alpha,
            "total"      : len(results),
            "success"    : success,
            "failed"     : failed,
            "output_file": output_file
        })

    except Exception as e:
        logger.error(f"Error /run_experiment: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/sbert-status", methods=["GET"])
def sbert_status():
    global sbert_model
    loaded = sbert_model is not None
    return jsonify({
        "model_loaded": loaded,
        "message": "✓ SBERT ready" if loaded else "✗ SBERT not loaded"
    })

@app.route("/init-sbert", methods=["POST"])
def init_sbert():
    try:
        get_sbert_model()
        return jsonify({"message": "✓ SBERT model ready!", "status": "ready", "model_loaded": True})
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500

@app.route("/clear-sbert", methods=["POST"])
def clear_sbert():
    clear_sbert_memory()
    return jsonify({"message": "✓ SBERT cleared from memory"})


if __name__ == "__main__":
    app.run(debug=False, threaded=True, port=5000, use_reloader=False)