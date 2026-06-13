# from streamlit import json

# import json
# import pytrec_eval
# import requests
# import copy
# from pathlib import Path

# API_BASE = "http://127.0.0.1:5000"

# # ════════════════════════════════════════════════════════════════
# # 1. QRELS — Ground Truth dari Ground Truth v2
# #    Format: {query: {candidate_title: relevance_score}}
# #    Nilai: 3=sangat relevan, 2=relevan, 1=sedikit relevan, 0=tidak relevan
# # ════════════════════════════════════════════════════════════════

# BASE_DIR   = Path(__file__).resolve().parent
# input_file = BASE_DIR / "../0.05Eksperimen.json"

# qrels_path = BASE_DIR / "v6ground_truth_scored.json"
# with open(qrels_path, "r", encoding="utf-8") as f:
#     qrels_raw = json.load(f)

# qrels = {}
# for item in qrels_raw:
#     q = item["query_text"]
#     qrels[q] = {}
#     for method_candidates in item["candidates"].values():
#         for c in method_candidates:
#             title = c["title"].lower()
#             qrels[q][title] = c["relevance_score"]

# def get_run_from_backend(query):
#     resp = requests.get(
#         f"{API_BASE}/compare_all_methods",
#         params={"query": query}
#     )
#     return resp.json()


# def build_run(method_results):
#     run_bm25       = {}
#     run_sbert      = {}
#     run_hybrid     = {}

#     for query, data in method_results.items():
#         # BM25
#         run_bm25[query] = {}
#         for item in data["bm25"]:
#             run_bm25[query][item["title"].lower()] = float(item.get("score_bm25", 0.0))

#         # SBERT
#         run_sbert[query] = {}
#         for item in data["sbert"]:
#             run_sbert[query][item["title"].lower()] = float(item.get("score_sbert", 0.0))

#         # Hybrid
#         run_hybrid[query] = {}
#         for item in data["hybrid"]:
#             run_hybrid[query][item["title"].lower()] = float(item.get("score_hybrid", 0.0))

#     return run_bm25, run_sbert, run_hybrid



# def hitung_ndcg(qrels, run, label):
#     # Lowercase semua key di qrels supaya match dengan run
#     qrels_lower = {
#         q: {doc.lower(): score for doc, score in docs.items()}
#         for q, docs in qrels.items()
#         if q in run  # hanya query yang ada di run
#     }
#     run_filtered = {q: run[q] for q in qrels_lower}

#     evaluator = pytrec_eval.RelevanceEvaluator(qrels_lower, ["ndcg_cut_10"])
#     results   = evaluator.evaluate(run_filtered)

#     print(f"\n{'='*65}")
#     print(f"  nDCG — {label}")
#     print(f"{'='*65}")
#     print(f"  {'Query':<25} {'nDCG@10':>10}")
#     print(f"  {'-'*47}")

#     total_10 = 0
#     for query, metrics in results.items():
#         s10 = metrics.get("ndcg_cut_10", 0)
#         total_10 += s10
#         print(f"  {query:<25} {s10:>10.4f}")

#     n = len(results)
#     print(f"  {'-'*47}")
#     print(f"  {'Rata-rata':<25} {total_10/n:>10.4f}")
#     print(f"{'='*65}")
#     return total_10 / n


# if __name__ == "__main__":
#     print("Mengambil hasil dari backend...")

#     method_results = {}
#     for query in qrels.keys():
#         print(f"  -> query: '{query}'")
#         method_results[query] = get_run_from_backend(query)

#     run_bm25, run_sbert, run_hybrid = build_run(method_results)

#     results_summary = {}
#     results_summary["Re-ranking BM25"]      = hitung_ndcg(qrels, run_bm25,       "Re-ranking BM25")
#     results_summary["Re-ranking SBERT"]     = hitung_ndcg(qrels, run_sbert,      "Re-ranking SBERT")
#     results_summary["Re-ranking Hybrid"]    = hitung_ndcg(qrels, run_hybrid,     "Re-ranking Hybrid")

#     print(f"\n{'='*65}")
#     print("  PERBANDINGAN SEMUA METODE")
#     print(f"{'='*65}")
#     print(f"  {'Metode':<25}{'nDCG@10':>10}")
#     print(f"  {'-'*47}")
#     for metode, (s10) in results_summary.items():
#         print(f"  {metode:<25} {s10:>10.4f}")
#     print(f"{'='*65}")

# import json
# import pytrec_eval
# from pathlib import Path

# BASE_DIR        = Path(__file__).resolve().parent
# eksperimen_path = BASE_DIR / "eksperimentResult0.80.json"
# qrels_path      = BASE_DIR / "v6ground_truth_scored.json"

# with open(qrels_path, "r", encoding="utf-8") as f:
#     qrels_raw = json.load(f)

# qrels = {}
# for item in qrels_raw:
#     q = item["query_text"]
#     if q not in qrels:
#         qrels[q] = {}
#     for method_candidates in item["candidates"].values():
#         for c in method_candidates:
#             title = c["title"].lower()
#             score = c.get("relevance_score", 0)
#             # Ambil score tertinggi kalau title muncul di beberapa method
#             if title not in qrels[q] or score > qrels[q][title]:
#                 qrels[q][title] = score

# with open(eksperimen_path, "r", encoding="utf-8") as f:
#     eksperimen_raw = json.load(f)

# def build_run(eksperimen_raw):
#     run_bm25       = {}
#     run_sbert      = {}
#     run_hybrid     = {}

#     for item in eksperimen_raw:
#         q          = item["query_text"]
#         candidates = item["candidates"]

#         # BM25
#         run_bm25[q] = {}
#         for c in candidates.get("bm25", []):
#             run_bm25[q][c["title"].lower()] = float(c.get("score_bm25", 0.0))

#         # SBERT
#         run_sbert[q] = {}
#         for c in candidates.get("sbert", []):
#             run_sbert[q][c["title"].lower()] = float(c.get("score_sbert", 0.0))

#         # Hybrid
#         run_hybrid[q] = {}
#         for c in candidates.get("hybrid", []):
#             run_hybrid[q][c["title"].lower()] = float(c.get("score_hybrid", 0.0))

#     return run_bm25, run_sbert, run_hybrid


# def hitung_ndcg(qrels, run, label):
#     qrels_filtered = {
#         q: {doc.lower(): score for doc, score in docs.items()}
#         for q, docs in qrels.items()
#         if q in run
#     }
#     run_filtered = {q: run[q] for q in qrels_filtered}

#     if not qrels_filtered:
#         print(f"\n[SKIP] {label} — tidak ada query yang cocok antara qrels dan run")
#         return 0.0

#     evaluator = pytrec_eval.RelevanceEvaluator(qrels_filtered, ["ndcg_cut_10"])
#     results   = evaluator.evaluate(run_filtered)

#     total = 0
#     for query, metrics in results.items():
#         s = metrics.get("ndcg_cut_10", 0)
#         total += s

#     n = len(results)
#     avg = total / n if n > 0 else 0
    
#     return avg


# if __name__ == "__main__":
#     print(f"\nMembaca eksperimen dari: {eksperimen_path}")
#     print(f"Jumlah entri eksperimen : {len(eksperimen_raw)}")
#     print(f"Jumlah query di qrels   : {len(qrels)}")

#     run_bm25, run_sbert, run_hybrid = build_run(eksperimen_raw)

#     results_summary = {}
#     results_summary["Re-ranking BM25"]   = hitung_ndcg(qrels, run_bm25,       "Re-ranking BM25")
#     results_summary["Re-ranking SBERT"]  = hitung_ndcg(qrels, run_sbert,      "Re-ranking SBERT")
#     results_summary["Re-ranking Hybrid"] = hitung_ndcg(qrels, run_hybrid,     "Re-ranking Hybrid")

#     print(f"\n\n  {'Metode':<25} {'nDCG@10':>10}")
#     print(f"  {'-'*47}")
#     for metode, avg in results_summary.items():
#         print(f"  {metode:<25} {avg:>10.4f}")
#     print(f"{'='*65}")

# import json
# import pytrec_eval
# from pathlib import Path
# import argparse

# BASE_DIR   = Path(__file__).resolve().parent
# qrels_path = BASE_DIR / "v6ground_truth_scored.json"

# parser = argparse.ArgumentParser()
# parser.add_argument("--file", required=True, help="Nama file eksperimen JSON")
# args   = parser.parse_args()

# eksperimen_path = BASE_DIR / args.file

# with open(qrels_path, "r", encoding="utf-8") as f:
#     qrels_raw = json.load(f)

# with open(eksperimen_path, "r", encoding="utf-8") as f:
#     eksperimen_raw = json.load(f)

# # Build qrels dari GT
# qrels = {}
# for item in qrels_raw:
#     q = item["query_text"]
#     if q not in qrels:
#         qrels[q] = {}
#     for method_candidates in item["candidates"].values():
#         for c in method_candidates:
#             title = c["title"].lower()
#             score = c.get("relevance_score", 0)
#             if title not in qrels[q] or score > qrels[q][title]:
#                 qrels[q][title] = score

# def build_run(eksperimen_raw):
#     run_bm25   = {}
#     run_sbert  = {}
#     run_hybrid = {}

#     for item in eksperimen_raw:
#         q     = item["query_text"]
#         cands = item["candidates"]

#         run_bm25[q] = {
#             c["title"].lower(): float(c.get("score_bm25", 0.0))
#             for c in cands.get("bm25", [])
#         }
#         run_sbert[q] = {
#             c["title"].lower(): float(c.get("score_sbert", 0.0))
#             for c in cands.get("sbert", [])
#         }
#         run_hybrid[q] = {
#             c["title"].lower(): float(c.get("score_hybrid", 0.0))
#             for c in cands.get("hybrid", [])
#         }

#     return run_bm25, run_sbert, run_hybrid

# def hitung_ndcg(qrels, run, label):
#     qrels_filtered = {q: d for q, d in qrels.items() if q in run}
#     run_filtered   = {q: run[q] for q in qrels_filtered}
#     if not qrels_filtered:
#         print(f"  [SKIP] {label}")
#         return 0.0
#     evaluator = pytrec_eval.RelevanceEvaluator(qrels_filtered, ["ndcg_cut_10"])
#     results   = evaluator.evaluate(run_filtered)
#     avg = sum(m["ndcg_cut_10"] for m in results.values()) / len(results)
#     return avg

# if __name__ == "__main__":
#     print(f"\nFile      : {eksperimen_path.name}")
#     print(f"Eksperimen: {len(eksperimen_raw)} query")
#     print(f"qrels     : {len(qrels)} query")

#     run_bm25, run_sbert, run_hybrid = build_run(eksperimen_raw)

#     print(f"\n  {'Metode':<28} {'nDCG@10':>8}")
#     print(f"  {'-'*40}")
#     for label, run in [
#         ("Re-ranking BM25",   run_bm25),
#         ("Re-ranking SBERT",  run_sbert),
#         ("Re-ranking Hybrid", run_hybrid),
#     ]:
#         avg = hitung_ndcg(qrels, run, label)
#         print(f"  {label:<28} {avg:>8.4f}")
#     print(f"  {'='*40}")

# from streamlit import json

# import json
# import pytrec_eval
# import requests
# import copy
# from pathlib import Path

# API_BASE = "http://127.0.0.1:5000"

# # ════════════════════════════════════════════════════════════════
# # 1. QRELS — Ground Truth dari Ground Truth v2
# #    Format: {query: {candidate_title: relevance_score}}
# #    Nilai: 3=sangat relevan, 2=relevan, 1=sedikit relevan, 0=tidak relevan
# # ════════════════════════════════════════════════════════════════

# BASE_DIR   = Path(__file__).resolve().parent
# input_file = BASE_DIR / "../0.05Eksperimen.json"

# qrels_path = BASE_DIR / "v6ground_truth_scored.json"
# with open(qrels_path, "r", encoding="utf-8") as f:
#     qrels_raw = json.load(f)

# qrels = {}
# for item in qrels_raw:
#     q = item["query_text"]
#     qrels[q] = {}
#     for method_candidates in item["candidates"].values():
#         for c in method_candidates:
#             title = c["title"].lower()
#             qrels[q][title] = c["relevance_score"]

# def get_run_from_backend(query):
#     resp = requests.get(
#         f"{API_BASE}/compare_all_methods",
#         params={"query": query}
#     )
#     return resp.json()


# def build_run(method_results):
#     run_bm25       = {}
#     run_sbert      = {}
#     run_hybrid     = {}

#     for query, data in method_results.items():
#         # BM25
#         run_bm25[query] = {}
#         for item in data["bm25"]:
#             run_bm25[query][item["title"].lower()] = float(item.get("score_bm25", 0.0))

#         # SBERT
#         run_sbert[query] = {}
#         for item in data["sbert"]:
#             run_sbert[query][item["title"].lower()] = float(item.get("score_sbert", 0.0))

#         # Hybrid
#         run_hybrid[query] = {}
#         for item in data["hybrid"]:
#             run_hybrid[query][item["title"].lower()] = float(item.get("score_hybrid", 0.0))

#     return run_bm25, run_sbert, run_hybrid



# def hitung_ndcg(qrels, run, label):
#     # Lowercase semua key di qrels supaya match dengan run
#     qrels_lower = {
#         q: {doc.lower(): score for doc, score in docs.items()}
#         for q, docs in qrels.items()
#         if q in run  # hanya query yang ada di run
#     }
#     run_filtered = {q: run[q] for q in qrels_lower}

#     evaluator = pytrec_eval.RelevanceEvaluator(qrels_lower, ["ndcg_cut_10"])
#     results   = evaluator.evaluate(run_filtered)

#     print(f"\n{'='*65}")
#     print(f"  nDCG — {label}")
#     print(f"{'='*65}")
#     print(f"  {'Query':<25} {'nDCG@10':>10}")
#     print(f"  {'-'*47}")

#     total_10 = 0
#     for query, metrics in results.items():
#         s10 = metrics.get("ndcg_cut_10", 0)
#         total_10 += s10
#         print(f"  {query:<25} {s10:>10.4f}")

#     n = len(results)
#     print(f"  {'-'*47}")
#     print(f"  {'Rata-rata':<25} {total_10/n:>10.4f}")
#     print(f"{'='*65}")
#     return total_10 / n


# if __name__ == "__main__":
#     print("Mengambil hasil dari backend...")

#     method_results = {}
#     for query in qrels.keys():
#         print(f"  -> query: '{query}'")
#         method_results[query] = get_run_from_backend(query)

#     run_bm25, run_sbert, run_hybrid = build_run(method_results)

#     results_summary = {}
#     results_summary["Re-ranking BM25"]      = hitung_ndcg(qrels, run_bm25,       "Re-ranking BM25")
#     results_summary["Re-ranking SBERT"]     = hitung_ndcg(qrels, run_sbert,      "Re-ranking SBERT")
#     results_summary["Re-ranking Hybrid"]    = hitung_ndcg(qrels, run_hybrid,     "Re-ranking Hybrid")

#     print(f"\n{'='*65}")
#     print("  PERBANDINGAN SEMUA METODE")
#     print(f"{'='*65}")
#     print(f"  {'Metode':<25}{'nDCG@10':>10}")
#     print(f"  {'-'*47}")
#     for metode, (s10) in results_summary.items():
#         print(f"  {metode:<25} {s10:>10.4f}")
#     print(f"{'='*65}")

# import json
# import pytrec_eval
# from pathlib import Path

# BASE_DIR        = Path(__file__).resolve().parent
# eksperimen_path = BASE_DIR / "eksperimentResult0.80.json"
# qrels_path      = BASE_DIR / "v6ground_truth_scored.json"

# with open(qrels_path, "r", encoding="utf-8") as f:
#     qrels_raw = json.load(f)

# qrels = {}
# for item in qrels_raw:
#     q = item["query_text"]
#     if q not in qrels:
#         qrels[q] = {}
#     for method_candidates in item["candidates"].values():
#         for c in method_candidates:
#             title = c["title"].lower()
#             score = c.get("relevance_score", 0)
#             # Ambil score tertinggi kalau title muncul di beberapa method
#             if title not in qrels[q] or score > qrels[q][title]:
#                 qrels[q][title] = score

# with open(eksperimen_path, "r", encoding="utf-8") as f:
#     eksperimen_raw = json.load(f)

# def build_run(eksperimen_raw):
#     run_bm25       = {}
#     run_sbert      = {}
#     run_hybrid     = {}

#     for item in eksperimen_raw:
#         q          = item["query_text"]
#         candidates = item["candidates"]

#         # BM25
#         run_bm25[q] = {}
#         for c in candidates.get("bm25", []):
#             run_bm25[q][c["title"].lower()] = float(c.get("score_bm25", 0.0))

#         # SBERT
#         run_sbert[q] = {}
#         for c in candidates.get("sbert", []):
#             run_sbert[q][c["title"].lower()] = float(c.get("score_sbert", 0.0))

#         # Hybrid
#         run_hybrid[q] = {}
#         for c in candidates.get("hybrid", []):
#             run_hybrid[q][c["title"].lower()] = float(c.get("score_hybrid", 0.0))

#     return run_bm25, run_sbert, run_hybrid


# def hitung_ndcg(qrels, run, label):
#     qrels_filtered = {
#         q: {doc.lower(): score for doc, score in docs.items()}
#         for q, docs in qrels.items()
#         if q in run
#     }
#     run_filtered = {q: run[q] for q in qrels_filtered}

#     if not qrels_filtered:
#         print(f"\n[SKIP] {label} — tidak ada query yang cocok antara qrels dan run")
#         return 0.0

#     evaluator = pytrec_eval.RelevanceEvaluator(qrels_filtered, ["ndcg_cut_10"])
#     results   = evaluator.evaluate(run_filtered)

#     total = 0
#     for query, metrics in results.items():
#         s = metrics.get("ndcg_cut_10", 0)
#         total += s

#     n = len(results)
#     avg = total / n if n > 0 else 0
    
#     return avg


# if __name__ == "__main__":
#     print(f"\nMembaca eksperimen dari: {eksperimen_path}")
#     print(f"Jumlah entri eksperimen : {len(eksperimen_raw)}")
#     print(f"Jumlah query di qrels   : {len(qrels)}")

#     run_bm25, run_sbert, run_hybrid = build_run(eksperimen_raw)

#     results_summary = {}
#     results_summary["Re-ranking BM25"]   = hitung_ndcg(qrels, run_bm25,       "Re-ranking BM25")
#     results_summary["Re-ranking SBERT"]  = hitung_ndcg(qrels, run_sbert,      "Re-ranking SBERT")
#     results_summary["Re-ranking Hybrid"] = hitung_ndcg(qrels, run_hybrid,     "Re-ranking Hybrid")

#     print(f"\n\n  {'Metode':<25} {'nDCG@10':>10}")
#     print(f"  {'-'*47}")
#     for metode, avg in results_summary.items():
#         print(f"  {metode:<25} {avg:>10.4f}")
#     print(f"{'='*65}")

import json
import pytrec_eval
from pathlib import Path
import argparse

BASE_DIR   = Path(__file__).resolve().parent
qrels_path = BASE_DIR / "v6ground_truth_scored.json"

parser = argparse.ArgumentParser()
parser.add_argument("--file", required=True, help="Nama file eksperimen JSON")
args   = parser.parse_args()

eksperimen_path = BASE_DIR / args.file

with open(qrels_path, "r", encoding="utf-8") as f:
    qrels_raw = json.load(f)

with open(eksperimen_path, "r", encoding="utf-8") as f:
    eksperimen_raw = json.load(f)

# Build qrels dari GT
qrels = {}
for item in qrels_raw:
    q = item["query_text"]
    if q not in qrels:
        qrels[q] = {}
    for method_candidates in item["candidates"].values():
        for c in method_candidates:
            title = c["title"].lower()
            score = c.get("relevance_score", 0)
            if title not in qrels[q] or score > qrels[q][title]:
                qrels[q][title] = score

# def build_run(eksperimen_raw):
#     run_bm25   = {}
#     run_sbert  = {}
#     run_hybrid = {}

#     for item in eksperimen_raw:
#         q     = item["query_text"]
#         cands = item["candidates"]

#         run_bm25[q] = {
#             c["title"].lower(): float(c.get("score_bm25", 0.0))
#             for c in cands.get("bm25", [])
#         }
#         run_sbert[q] = {
#             c["title"].lower(): float(c.get("score_sbert", 0.0))
#             for c in cands.get("sbert", [])
#         }
#         run_hybrid[q] = {
#             c["title"].lower(): float(c.get("score_hybrid", 0.0))
#             for c in cands.get("hybrid", [])
#         }

#     return run_bm25, run_sbert, run_hybrid
def build_run(eksperimen_raw):
    run_bm25   = {}
    run_sbert  = {}
    run_hybrid = {}

    for item in eksperimen_raw:
        q     = item["query_text"]
        cands = item["candidates"]

        # BM25
        run_bm25[q] = {}
        for c in cands.get("bm25", []):
            title = c["title"].lower()
            try:
                # Ambil score, ganti ke 0.0 jika berupa None atau string kosong
                val = c.get("score_bm25")
                score = float(val) if val is not None and val != "" else 0.0
            except ValueError:
                score = 0.0
            
            # Jika ada judul duplikat di satu query, ambil skor tertinggi agar stabil
            if title not in run_bm25[q] or score > run_bm25[q][title]:
                run_bm25[q][title] = score

        # SBERT
        run_sbert[q] = {}
        for c in cands.get("sbert", []):
            title = c["title"].lower()
            try:
                val = c.get("score_sbert")
                score = float(val) if val is not None and val != "" else 0.0
            except ValueError:
                score = 0.0
            if title not in run_sbert[q] or score > run_sbert[q][title]:
                run_sbert[q][title] = score

        # Hybrid
        run_hybrid[q] = {}
        for c in cands.get("hybrid", []):
            title = c["title"].lower()
            try:
                val = c.get("score_hybrid")
                score = float(val) if val is not None and val != "" else 0.0
            except ValueError:
                score = 0.0
            if title not in run_hybrid[q] or score > run_hybrid[q][title]:
                run_hybrid[q][title] = score

    return run_bm25, run_sbert, run_hybrid

def hitung_ndcg(qrels, run, label):
    qrels_filtered = {q: d for q, d in qrels.items() if q in run}
    run_filtered   = {q: run[q] for q in qrels_filtered}
    if not qrels_filtered:
        print(f"  [SKIP] {label}")
        return 0.0
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_filtered, ["ndcg_cut_10"])
    results   = evaluator.evaluate(run_filtered)
    avg = sum(m["ndcg_cut_10"] for m in results.values()) / len(results)
    return avg

if __name__ == "__main__":
    print(f"\nFile      : {eksperimen_path.name}")
    print(f"Eksperimen: {len(eksperimen_raw)} query")
    print(f"qrels     : {len(qrels)} query")

    run_bm25, run_sbert, run_hybrid = build_run(eksperimen_raw)

    print(f"\n  {'Metode':<28} {'nDCG@10':>8}")
    print(f"  {'-'*40}")
    for label, run_data in [
        ("Re-ranking BM25",   run_bm25),
        ("Re-ranking SBERT",  run_sbert),
        ("Re-ranking Hybrid", run_hybrid),
    ]:
        avg = hitung_ndcg(qrels, run_data, label)
        print(f"  {label:<28} {avg:>8.4f}")
        # print(json.dumps(
        #     dict(list(qrels.items())[:3]),
        #     indent=2,
        #     ensure_ascii=False
        # ))
    print(f"  {'='*40}")