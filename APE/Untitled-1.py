"""
experiment.py
=============
Otomasi eksperimen reranking search suggestion buku.

Cara pakai:
    python experiment.py --alpha 0.60
    python experiment.py --alpha 0.65
    python experiment.py --alpha 0.70
    python experiment.py --alpha 0.75
    python experiment.py --alpha 0.80

Output per run:
    experiment_results_alpha{xx}.json  — data mentah
    experiment_viz_alpha{xx}.html      — visualisasi interaktif
"""

import argparse, copy, json, logging, os
from pathlib import Path

from opensearchpy import OpenSearch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
INDEX_NAME  = "books2"
BASE_DIR    = Path(__file__).resolve().parent
INPUT_FILE  = BASE_DIR / "../../DATASET/test_queries_final.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# OPENSEARCH
# ─────────────────────────────────────────────────────────────────
client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_auth=("admin", "KoTA404TABAH!"),
    use_ssl=True, verify_certs=False,
    ssl_show_warn=False, timeout=60,
    max_retries=3, retry_on_timeout=True
)

# ─────────────────────────────────────────────────────────────────
# SBERT (lazy load)
# ─────────────────────────────────────────────────────────────────
_sbert = None
def sbert():
    global _sbert
    if _sbert is None:
        log.info("Loading SBERT...")
        _sbert = SentenceTransformer("all-MiniLM-L6-v2")
        log.info("✓ SBERT ready")
    return _sbert

# ─────────────────────────────────────────────────────────────────
# CANDIDATE GENERATION
# ─────────────────────────────────────────────────────────────────
def get_candidates(query: str) -> list[dict]:
    """
    Ambil 10 kandidat dari OpenSearch.
    Prioritas: completion suggester → fallback fuzzy.
    """
    body = {
        "suggest": {
            "by_title":  {"prefix": query, "completion": {"field": "suggest_title",  "size": 10, "skip_duplicates": True}},
            "by_author": {"prefix": query, "completion": {"field": "suggest_author", "size": 10, "skip_duplicates": True}},
        },
        "query": {"bool": {"should": [
            {"match": {"title":  {"query": query, "fuzziness": "2"}}},
            {"match": {"author": {"query": query, "fuzziness": "2"}}},
        ], "minimum_should_match": "1"}},
        "size": 10,
        "_source": ["title", "author", "description", "numRatings"],
    }

    res  = client.search(index=INDEX_NAME, body=body)
    seen = set()
    out  = []

    def add(src, source_label):
        title = src.get("title", "")
        if not title or title in seen or len(out) >= 10:
            return
        seen.add(title)
        out.append({
            "title":       title,
            "author":      src.get("author", ""),
            "description": src.get("description", ""),
            "numRatings":  int(src.get("numRatings", 0)),
            "source":      source_label,
        })

    for key in ["by_title", "by_author"]:
        for opt in res["suggest"][key][0]["options"]:
            add(opt["_source"], "completion")

    if len(out) < 10:
        for hit in res.get("hits", {}).get("hits", []):
            add(hit["_source"], "fuzzy")

    return out

# ─────────────────────────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────────────────────────
def bm25_scores(query, candidates):
    corpus  = [f"{c['title']} {c['author']}" for c in candidates]
    bm25    = BM25Okapi([d.lower().split() for d in corpus])
    return bm25.get_scores(query.lower().split())

def sbert_scores(query, candidates):
    corpus  = [f"{c['title']} {c['author']} {c['description']}" for c in candidates]
    qvec    = sbert().encode([query])
    cvec    = sbert().encode(corpus)
    return cosine_similarity(qvec, cvec)[0]

def minmax(scores):
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]

# ─────────────────────────────────────────────────────────────────
# RERANKING
# ─────────────────────────────────────────────────────────────────
def rank_bm25(query, cands):
    sc = bm25_scores(query, cands)
    for i, c in enumerate(cands):
        c["score_bm25"] = float(sc[i])
    return sorted(cands, key=lambda x: x["score_bm25"], reverse=True)

def rank_sbert(query, cands):
    sc = sbert_scores(query, cands)
    for i, c in enumerate(cands):
        c["score_sbert"] = float(sc[i])
    return sorted(cands, key=lambda x: x["score_sbert"], reverse=True)

def rank_hybrid(query, cands, alpha):
    """score = alpha * SBERT_norm + (1-alpha) * BM25_norm"""
    b = minmax(list(bm25_scores(query, cands)))
    s = minmax([float(x) for x in sbert_scores(query, cands)])
    for i, c in enumerate(cands):
        c["score_bm25"]   = float(bm25_scores(query, cands)[i])
        c["score_sbert"]  = float(sbert_scores(query, cands)[i])
        c["score_hybrid"] = float((1 - alpha) * b[i] + alpha * s[i])
    return sorted(cands, key=lambda x: x["score_hybrid"], reverse=True)

def strip_desc(lists):
    for lst in lists:
        for c in lst:
            c.pop("description", None)

# ─────────────────────────────────────────────────────────────────
# EKSPERIMEN
# ─────────────────────────────────────────────────────────────────
def run(alpha: float):
    if not INPUT_FILE.exists():
        log.error(f"File tidak ditemukan: {INPUT_FILE}")
        return

    queries = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    tag     = f"{alpha:.2f}".replace(".", "")
    out_json = BASE_DIR / f"experiment_results_alpha{tag}.json"
    out_html = BASE_DIR / f"experiment_viz_alpha{tag}.html"

    log.info(f"▶ Eksperimen — {len(queries)} query, alpha={alpha}")
    results = []

    for i, q in enumerate(queries, 1):
        qt = q["query_text"]
        log.info(f"  [{i:3}/{len(queries)}] {q['query_id']} | {q['query_type']:15} | \"{qt}\"")

        cands = get_candidates(qt)
        if not cands:
            results.append({**_base(q, alpha), "status": "FAILED",
                             "candidates": {"completion": [], "bm25": [], "sbert": [], "hybrid": []}})
            continue

        comp   = copy.deepcopy(cands)
        ranked_bm25   = rank_bm25(qt,   copy.deepcopy(cands))
        ranked_sbert  = rank_sbert(qt,  copy.deepcopy(cands))
        ranked_hybrid = rank_hybrid(qt, copy.deepcopy(cands), alpha)
        strip_desc([comp, ranked_bm25, ranked_sbert, ranked_hybrid])

        results.append({**_base(q, alpha), "status": "OK",
                        "candidates": {"completion": comp,
                                       "bm25": ranked_bm25,
                                       "sbert": ranked_sbert,
                                       "hybrid": ranked_hybrid}})

    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    ok = sum(1 for r in results if r["status"] == "OK")
    log.info(f"✓ JSON  → {out_json}  (OK:{ok} | Failed:{len(results)-ok})")

    out_html.write_text(_build_html(results, alpha), encoding="utf-8")
    log.info(f"✓ HTML  → {out_html}")
    log.info(f"  Buka di browser: file:///{out_html.resolve()}")

def _base(q, alpha):
    return {"query_id": q["query_id"], "query_text": q["query_text"],
            "query_type": q["query_type"], "title": q.get("title",""),
            "author": q.get("author",""), "alpha": alpha}

# ─────────────────────────────────────────────────────────────────
# HTML VISUALISASI
# ─────────────────────────────────────────────────────────────────
def _build_html(results: list, alpha: float) -> str:
    data  = json.dumps(results, ensure_ascii=False)
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
const D=JSON.parse({repr(data)});
let F=[...D],idx=0;

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


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Bobot SBERT dalam hybrid (default 0.7)")
    alpha = parser.parse_args().alpha

    log.info("⏳ Loading SBERT...")
    sbert()
    run(alpha)