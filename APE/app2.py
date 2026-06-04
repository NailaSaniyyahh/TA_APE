from opensearchpy import OpenSearch
from rank_bm25 import BM25Okapi
import copy, json, os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from pathlib import Path
import argparse, copy, json, logging, os


INDEX_NAME  = "books2"
BASE_DIR    = Path(__file__).resolve().parent
input_file  = BASE_DIR / "../CODE DATASET/v3test_queries_final.json"
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)


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

sbert_model = None

def get_sbert_model():
    global sbert_model
    if sbert_model is None:
        logger.info("📦 Loading SBERT model...")
        sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("✓ SBERT model loaded")
    return sbert_model

# def clear_sbert_memory():
#     global sbert_model
#     if sbert_model is not None:
#         import gc
#         del sbert_model
#         sbert_model = None
#         gc.collect()
#         logger.info("🧹 SBERT cleared from memory")

# logger.info("⏳ Pre-loading SBERT...")
# get_sbert_model()
# logger.info("✓ SBERT ready!")

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
                "should": [
                    {"match": {"title":  {"query": query, "fuzziness": "2"}}},
                    {"match": {"author": {"query": query, "fuzziness": "2"}}}
                ],
                "minimum_should_match": "1"
            }
        },
        "size": 10,
        "_source": ["title", "author", "description", "numRatings"]
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
                "source":      "fuzzy"
            })

    return result

def compute_bm25_scores(query, candidates):
    corpus           = [f"{c['title']} {c['author']}" for c in candidates]
    bm25             = BM25Okapi(doc.lower().split() for doc in corpus)
    return bm25.get_scores(query.lower().split())

def compute_sbert_scores(query, candidates):
    corpus        = [f"{c['title']} {c['author']} {c['description']}" for c in candidates]
    query_vec     = get_sbert_model().encode([query])
    candidate_vec = get_sbert_model().encode(corpus)
    return cosine_similarity(query_vec, candidate_vec)[0]

def normalize(scores):
    min_s, max_s = min(scores), max(scores)
    if max_s - min_s == 0:
        return [0.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]

def rerank_bm25(query, candidates):
    scores = compute_bm25_scores(query, candidates)
    for i, c in enumerate(candidates):
        c["score_bm25"] = float(scores[i])
    return sorted(candidates, key=lambda x: x["score_bm25"], reverse=True)

def rerank_sbert(query, candidates):
    scores = compute_sbert_scores(query, candidates)
    for i, c in enumerate(candidates):
        c["score_sbert"] = float(scores[i])
    return sorted(candidates, key=lambda x: x["score_sbert"], reverse=True)

def rerank_hybrid(query, candidates, alpha):
    # score = alpha * SBERT_norm + (1-alpha) * BM25_norm
    bm25_scores  = compute_bm25_scores(query, candidates)
    sbert_scores = compute_sbert_scores(query, candidates)
    bm25_norm    = normalize(list(bm25_scores))
    sbert_norm   = normalize([float(s) for s in sbert_scores])

    for i, c in enumerate(candidates):
        c["score_bm25"]   = float(bm25_scores[i])
        c["score_sbert"]  = float(sbert_scores[i])
        c["score_hybrid"] = float((1 - alpha) * bm25_norm[i] + alpha * sbert_norm[i])

    return sorted(candidates, key=lambda x: x["score_hybrid"], reverse=True)

def strip_desc(lists):
    for lst in lists:
        for c in lst:
            c.pop("description", None)


# eksperimen
def run(alpha: float):
    if not input_file.exists():
        logger.error(f"File tidak ditemukan: {input_file}")
        return

    queries = json.loads(input_file.read_text(encoding="utf-8"))
    tag     = f"{alpha:.2f}".replace(".", "")
    out_json = BASE_DIR / f"v5experiment_results_alpha{tag}.json"
    out_html = BASE_DIR / f"v5experiment_viz_alpha{tag}.html"

    logger.info(f" Eksperimen — {len(queries)} query, alpha={alpha}")
    results = []

    for i, q in enumerate(queries, 1):
        qtext = q["query_text"]
        logger.info(f"[{i:3}/{len(queries)}] Processing: '{qtext}'")

        candidates = get_candidates(qtext)
        if not candidates:
            results.append({**_base(q, alpha), "status": "FAILED",
                             "candidates": {"completion": [], "bm25": [], "sbert": [], "hybrid": []}})
            continue
        
        completion_order = copy.deepcopy(candidates)
        bm25       = rerank_bm25(qtext,   copy.deepcopy(candidates))
        sbert      = rerank_sbert(qtext,  copy.deepcopy(candidates))
        hybrid     = rerank_hybrid(qtext, copy.deepcopy(candidates), alpha=alpha)
        strip_desc([completion_order, bm25, sbert, hybrid])

        results.append({**_base(q, alpha), "status": "OK",
                         "candidates": {
                             "completion": completion_order,
                             "bm25":       bm25,
                             "sbert":      sbert,
                             "hybrid":     hybrid
                         }})

    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    ok = sum(1 for r in results if r["status"] == "OK")
    logger.info(f"✓ JSON  → {out_json}  (OK:{ok} | Failed:{len(results)-ok})")

    out_html.write_text(
        _build_html(out_json.name, alpha),
        encoding="utf-8"
    )
    logger.info(f"✓ HTML  → {out_html}")
    logger.info(f"  Buka di browser: file:///{out_html.resolve()}")

def _base(q, alpha):
    return {"query_id": q["query_id"], "query_text": q["query_text"],
            "query_type": q["query_type"], "title": q.get("title",""),
            "author": q.get("author",""), "alpha": alpha}


# visaisasi html
def _build_html(json_filename: str, alpha: float) -> str:
    a_str = f"{alpha:.2f}"
    return f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<title>Experiment — alpha={a_str}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',sans-serif;background:#f4f6fb;padding:32px 16px;color:#1a1a2e}}
.wrap{{max-width:1200px;margin:0 auto}}
h1{{font-size:22px;margin-bottom:4px}}
.sub{{color:#888;font-size:13px;margin-bottom:24px}}
/* nav */
.nav{{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:16px}}
.btn{{padding:7px 16px;border-radius:8px;border:1.5px solid #ddd;background:#fff;
      cursor:pointer;font-size:13px;color:#555;font-family:inherit;transition:.15s}}
.btn:hover{{background:#eef2ff;border-color:#4c6ef5;color:#4c6ef5}}
.btn:disabled{{opacity:.3;cursor:not-allowed}}
.counter{{font-size:13px;color:#888;padding:7px 14px;background:#fff;
          border:1.5px solid #eee;border-radius:8px}}
/* filter chips */
.chips{{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:16px}}
.chip{{padding:5px 13px;border-radius:20px;border:1.5px solid #ddd;
       background:#fff;cursor:pointer;font-size:11px;color:#888;font-family:inherit;transition:.15s}}
.chip.on{{background:#1a1a2e;color:#fff;border-color:#1a1a2e}}
/* search box */
.sbox{{display:flex;align-items:center;gap:10px;background:#fff;
       border:1.5px solid #ddd;border-radius:12px;padding:13px 18px;
       box-shadow:0 2px 10px rgba(0,0,0,.06);margin-bottom:8px}}
.sbox input{{flex:1;border:none;outline:none;font-size:15px;font-family:inherit;color:#1a1a2e}}
.qmeta{{display:flex;gap:8px;align-items:center;font-size:11px;color:#aaa;
        padding:0 4px;margin-bottom:20px;flex-wrap:wrap}}
/* type badges */
.tb{{font-size:10px;padding:2px 8px;border-radius:10px;font-weight:700}}
.tb-PREFIX_TITLE{{background:#eef0ff;color:#4c6ef5}}
.tb-PREFIX_AUTHOR{{background:#fff4e6;color:#e67700}}
.tb-PARTIAL{{background:#e6fcf5;color:#0ca678}}
.tb-TYPO{{background:#fff0f6;color:#c2255c}}
/* grid */
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}
.col{{background:#fff;border-radius:14px;box-shadow:0 2px 12px rgba(0,0,0,.06);overflow:hidden}}
.col-hd{{padding:11px 16px;font-size:11px;font-weight:700;letter-spacing:.4px;border-bottom:1px solid #f0f0f0}}
.hd-c{{background:#eef2ff;color:#4c6ef5}}
.hd-b{{background:#ebfbee;color:#2f9e44}}
.hd-s{{background:#fff9db;color:#e67700}}
.hd-h{{background:#fff0f6;color:#c2255c}}
/* book item */
.item{{display:flex;align-items:flex-start;gap:8px;padding:10px 14px;
       border-bottom:1px solid #f7f7f7;transition:.1s}}
.item:last-child{{border:none}}
.item:hover{{background:#fafafa}}
.item.hit{{background:#f0fdf4}}
.rn{{font-size:11px;font-weight:700;color:#ccc;min-width:16px;padding-top:2px;flex-shrink:0}}
.item.hit .rn{{color:#2f9e44}}
.info{{flex:1;min-width:0}}
.ttl{{font-size:12px;font-weight:600;color:#1a1a2e;word-break:break-word;line-height:1.4}}
.item.hit .ttl{{color:#2f9e44}}
.aut{{font-size:10px;color:#aaa;margin-top:2px}}
.sc{{font-size:10px;color:#ccc;margin-top:2px}}
.right{{display:flex;flex-direction:column;align-items:flex-end;gap:3px;flex-shrink:0}}
.hb{{font-size:9px;background:#e6fcf5;color:#0ca678;padding:1px 6px;border-radius:6px;font-weight:700}}
.mv{{font-size:10px;padding:1px 6px;border-radius:8px;font-weight:600}}
.mv-up{{background:#e6fcf5;color:#0ca678}}
.mv-dn{{background:#fff5f5;color:#e03131}}
.mv-eq{{background:#f8f9fa;color:#aaa}}
.mv-nw{{background:#eef2ff;color:#4c6ef5}}
.src{{font-size:9px;padding:1px 5px;border-radius:4px}}
.src-completion{{background:#eef0ff;color:#4c6ef5}}
.src-fuzzy{{background:#fff4e6;color:#e67700}}
.miss{{padding:10px 14px;font-size:11px;color:#e03131;font-style:italic;border-top:1px dashed #ffe3e3}}
.empty{{padding:20px;text-align:center;color:#ccc;font-size:12px}}
</style>
</head>
<body>
<div class="wrap">
  <h1>📚 Experiment Results</h1>
  <p class="sub">alpha = {a_str} &nbsp;·&nbsp; Gunakan filter dan tombol navigasi untuk menelusuri setiap query.</p>

  <div class="nav">
    <button class="btn" id="bPrev" onclick="go(-1)">◀ Prev</button>
    <span class="counter" id="ctr"></span>
    <button class="btn" id="bNext" onclick="go(1)">Next ▶</button>
  </div>

  <div class="chips">
    <button class="chip on"  onclick="filt('ALL',this)">Semua</button>
    <button class="chip"     onclick="filt('PREFIX_TITLE',this)">PREFIX_TITLE</button>
    <button class="chip"     onclick="filt('PREFIX_AUTHOR',this)">PREFIX_AUTHOR</button>
    <button class="chip"     onclick="filt('PARTIAL',this)">PARTIAL</button>
    <button class="chip"     onclick="filt('TYPO',this)">TYPO</button>
  </div>

  <div class="sbox"><span>🔍</span><input id="qInput" readonly></div>
  <div class="qmeta">
    <span id="qBadge"></span>
    <span>Expected: <strong id="qExp"></strong></span>
    <span id="qAuth"></span>
  </div>

  <div class="grid">
    <div class="col"><div class="col-hd hd-c">Completion Suggester</div><div id="cC"></div></div>
    <div class="col"><div class="col-hd hd-b">BM25</div><div id="cB"></div></div>
    <div class="col"><div class="col-hd hd-s">SBERT</div><div id="cS"></div></div>
    <div class="col"><div class="col-hd hd-h">Hybrid (α={a_str})</div><div id="cH"></div></div>
  </div>
</div>

<script>
let D = [];
let F = [];
let idx = 0;

async function loadData() {{
try {{
const resp = await fetch("{json_filename}");

if (!resp.ok) {{
    throw new Error(`HTTP ${{resp.status}}`);
}}

D = await resp.json();
F = [...D];

render();
}} catch (err) }}

loadData();

function filt(t,btn){{
  document.querySelectorAll('.chip').forEach(b=>b.classList.remove('on'));
  btn.classList.add('on');
  F=t==='ALL'?[...D]:D.filter(q=>q.query_type===t);
  idx=0;render();
}}
function go(d){{idx=Math.max(0,Math.min(F.length-1,idx+d));render();}}

function render(){{
  if(!F.length){{
    ['cC','cB','cS','cH'].forEach(id=>{{document.getElementById(id).innerHTML='<div class="empty">Tidak ada query.</div>';}});
    document.getElementById('ctr').textContent='0/0'; return;
  }}
  const q=F[idx], exp=q.title.toLowerCase().trim();
  document.getElementById('qInput').value=q.query_text;
  document.getElementById('ctr').textContent=`${{q.query_id}} — ${{idx+1}}/${{F.length}}`;
  document.getElementById('qBadge').innerHTML=`<span class="tb tb-${{q.query_type}}">${{q.query_type}}</span>`;
  document.getElementById('qExp').textContent=q.title;
  document.getElementById('qAuth').textContent=q.author?`✍️ ${{q.author}}`:'';
  document.getElementById('bPrev').disabled=idx===0;
  document.getElementById('bNext').disabled=idx===F.length-1;

  const comp=q.candidates.completion||[];
  const rm={{}};comp.forEach((c,i)=>{{rm[c.title]=i+1;}});
  buildCol('cC',comp,'c',exp,rm);
  buildCol('cB',q.candidates.bm25||[],'b',exp,rm);
  buildCol('cS',q.candidates.sbert||[],'s',exp,rm);
  buildCol('cH',q.candidates.hybrid||[],'h',exp,rm);
}}

function buildCol(id,list,m,exp,rm){{
  const el=document.getElementById(id);
  if(!list.length){{el.innerHTML='<div class="empty">Belum ada data.</div>';return;}}
  let h=list.map((c,i)=>{{
    const hit=c.title.toLowerCase().trim()===exp;
    const sc=m==='b'?c.score_bm25:m==='s'?c.score_sbert:m==='h'?c.score_hybrid:null;
    const mv=m==='c'?'':(()=>{{
      const prev=rm[c.title];
      if(prev==null) return '<span class="mv mv-nw">baru</span>';
      const d=prev-(i+1);
      if(d>0) return `<span class="mv mv-up">▲${{d}}</span>`;
      if(d<0) return `<span class="mv mv-dn">▼${{Math.abs(d)}}</span>`;
      return '<span class="mv mv-eq">—</span>';
    }})();
    return `<div class="item ${{hit?'hit':''}}">
      <div class="rn">${{i+1}}</div>
      <div class="info">
        <div class="ttl">${{esc(c.title)}}</div>
        <div class="aut">${{esc(c.author||'')}}</div>
        ${{sc!=null?`<div class="sc">${{sc.toFixed(4)}}</div>`:''}}
      </div>
      <div class="right">
        ${{hit?'<span class="hb">✓ FOUND</span>':''}}
        ${{mv}}
        <span class="src src-${{c.source||'fuzzy'}}">${{c.source||''}}</span>
      </div>
    </div>`;
  }}).join('');
  if(!list.find(c=>c.title.toLowerCase().trim()===exp))
    h+='<div class="miss">⚠ Expected tidak ditemukan di top-10</div>';
  el.innerHTML=h;
}}

function esc(t){{return String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}}
render();
</script>
</body>
</html>"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Bobot SBERT dalam hybrid (default 0.7)")
    alpha = parser.parse_args().alpha

    logger.info("⏳ Loading SBERT...")
    get_sbert_model()
    run(alpha)
    
# @app.route("/compare_all_methods", methods=["GET"])
# def compare_all_methods():
#     """
#     GET /compare_all_methods?query=hunger&alpha=0.7

#     Dipakai oleh:
#     - index.html : visualisasi 4 kolom (alpha default 0.7)
#     - /run_experiment : otomasi eksperimen batch
#     """
#     try:
#         query     = request.args.get("query", "").strip()
#         alpha     = float(request.args.get("alpha", 0.7))
#         pop_boost = float(request.args.get("pop_boost", 0.0))

#         if not query:
#             return jsonify({"completion": [], "bm25": [], "sbert": [], "hybrid": []})

#         candidates = get_candidates(query)
#         if not candidates:
#             return jsonify({"completion": [], "bm25": [], "sbert": [], "hybrid": []})

#         completion_order = copy.deepcopy(candidates)
#         bm25_order       = rerank_bm25(query,   copy.deepcopy(candidates))
#         sbert_order      = rerank_sbert(query,  copy.deepcopy(candidates))
#         hybrid_order     = rerank_hybrid(query, copy.deepcopy(candidates), alpha=alpha, pop_boost=pop_boost)

#         # Hapus description dari response (tidak perlu di frontend)
#         for lst in [completion_order, bm25_order, sbert_order, hybrid_order]:
#             for c in lst:
#                 c.pop("description", None)

#         return jsonify({
#             "completion": completion_order,
#             "bm25":       bm25_order,
#             "sbert":      sbert_order,
#             "hybrid":     hybrid_order
#         })

#     except Exception as e:
#         logger.error(f"Error /compare_all_methods: {e}")
#         return jsonify({"error": str(e)}), 500


# @app.route("/run_experiment", methods=["GET"])
# def run_experiment():
#     """
#     GET /run_experiment?alpha=0.60

#     Endpoint otomasi eksperimen — menggantikan candidate_generation.py.
#     Membaca test_queries_final.json, jalankan semua query dengan
#     alpha yang diberikan, simpan hasilnya ke JSON.

#     Cara pakai:
#       http://localhost:5000/run_experiment?alpha=0.60
#       http://localhost:5000/run_experiment?alpha=0.65
#       http://localhost:5000/run_experiment?alpha=0.70
#       http://localhost:5000/run_experiment?alpha=0.75
#       http://localhost:5000/run_experiment?alpha=0.80
#     """
#     try:
#         alpha      = float(request.args.get("alpha", 0.7))
#         BASE_DIR   = Path(__file__).resolve().parent
#         input_file = BASE_DIR / "../../DATASET/v3test_queries_final.json"
#         # input_file = "../../test_queries_final.json"

#         if not os.path.exists(input_file):
#             return jsonify({"error": f"File {input_file} tidak ditemukan"}), 404

#         with open(input_file, "r", encoding="utf-8") as f:
#             queries = json.load(f)

#         alpha_str   = f"{alpha:.2f}".replace(".", "")
#         output_file = f"v4experiment_results_alpha{alpha_str}.json"

#         results = []
#         for i, q in enumerate(queries, 1):
#             qtext = q["query_text"]
#             logger.info(f"[{i:3}/{len(queries)}] Processing: '{qtext}'")

#             candidates = get_candidates(qtext)

#             if not candidates:
#                 results.append({
#                     "query_id"   : q["query_id"],
#                     "query_text" : qtext,
#                     "query_type" : q["query_type"],
#                     "title"      : q.get("title", ""),
#                     "author"     : q.get("author", ""),
#                     "alpha"      : alpha,
#                     "status"     : "FAILED",
#                     "candidates" : {
#                         "completion": []
#                     }
#                 })
#                 continue

#             completion_order = copy.deepcopy(candidates)
#             # bm25_order       = rerank_bm25(qtext,   copy.deepcopy(candidates))
#             # sbert_order      = rerank_sbert(qtext,  copy.deepcopy(candidates))
#             # hybrid_order     = rerank_hybrid(qtext, copy.deepcopy(candidates), alpha=alpha)

#             # Hapus description dari hasil
#             for lst in [completion_order]:
#                 for c in lst:
#                     c.pop("description", None)

#             results.append({
#                 "query_id"   : q["query_id"],
#                 "query_text" : qtext,
#                 "query_type" : q["query_type"],
#                 "title"      : q.get("title", ""),
#                 "author"     : q.get("author", ""),
#                 "alpha"      : alpha,
#                 "status"     : "OK",
#                 "candidates" : {
#                     "completion": completion_order
#                 }
#             })

#         with open(output_file, "w", encoding="utf-8") as f:
#             json.dump(results, f, indent=2, ensure_ascii=False)

#         success = sum(1 for r in results if r["status"] == "OK")
#         failed  = len(results) - success

#         logger.info(f"✓ Eksperimen selesai → {output_file}")
#         return jsonify({
#             "message"    : f"Eksperimen selesai dengan alpha={alpha}",
#             "alpha"      : alpha,
#             "total"      : len(results),
#             "success"    : success,
#             "failed"     : failed,
#             "output_file": output_file
#         })

#     except Exception as e:
#         logger.error(f"Error /run_experiment: {e}")
#         return jsonify({"error": str(e)}), 500


# @app.route("/health", methods=["GET"])
# def health():
#     return jsonify({"status": "ok"})

# @app.route("/sbert-status", methods=["GET"])
# def sbert_status():
#     global sbert_model
#     loaded = sbert_model is not None
#     return jsonify({
#         "model_loaded": loaded,
#         "message": "✓ SBERT ready" if loaded else "✗ SBERT not loaded"
#     })

# @app.route("/init-sbert", methods=["POST"])
# def init_sbert():
#     try:
#         get_sbert_model()
#         return jsonify({"message": "✓ SBERT model ready!", "status": "ready", "model_loaded": True})
#     except Exception as e:
#         return jsonify({"error": str(e), "status": "failed"}), 500

# @app.route("/clear-sbert", methods=["POST"])
# def clear_sbert():
#     clear_sbert_memory()
#     return jsonify({"message": "✓ SBERT cleared from memory"})


# if __name__ == "__main__":
#     app.run(debug=False, threaded=True, port=5000, use_reloader=False)