import pytrec_eval
import requests
import copy

API_BASE = "http://127.0.0.1:5000"

# ════════════════════════════════════════════════════════════════
# 1. QRELS — Ground Truth dari Ground Truth v2
#    Format: {query: {candidate_title: relevance_score}}
#    Nilai: 3=sangat relevan, 2=relevan, 1=sedikit relevan, 0=tidak relevan
# ════════════════════════════════════════════════════════════════
qrels = {
    "the hunger": {
        "the hunger angel": 1,
        "the hunger but mainly death games a parody": 0,
        "the hunger games": 3,
        "the hunger games official illustrated movie companion": 2,
        "the hunger games tribute guide": 2,
        "the hunger games trilogy boxset": 2,
    },
    "pride and": {
        "pride and pleasure": 1,
        "pride and prejudice": 3,
        "pride and prejudice and zombies": 2,
        "pride and prejudice and zombies the graphic novel": 1,
        "pride and prejudice ii the sequel": 1,
        "pride and prejudice retold in limericks": 0,
        "pride and prejudice the wild and wanton edition": 0,
    },
    "gone with": {
        "gone with the night the rape slaying trial": 0,
        "gone with the wind": 3,
        "gone with the wind letters": 2,
        "gone with the windsors": 1,
    },
    "the book": {
        "the book about blanche and marie": 1,
        "the book of 1 ariel": 0,
        "the book of atrix wolfe": 1,
        "the book of atrus": 1,
        "the book of awesome": 2,
        "the book of basketball the nba according to sports guy": 1,
        "the book of blood and shadow": 1,
        "the book of blots": 0,
        "the book of bunny suicides": 1,
        "the book thief": 3,
    },
    "the hobbit": {
        "the hobbit graphic novel": 2,
        "the hobbit or there and back again": 3,
        "the hobbit part one": 2,
    },
    "george or": {
        "animal farm 1984": 3,
        "down and out in paris and london": 2,
        "homage to catalonia": 2,
        "keep the aspidistra flying": 2,
        "burmese days": 2,
        "coming up for air": 2,
        "animal farm": 3,
        "1984": 3,
        "a collection of essays": 1,
        "george orwell omnibus the complete novels": 2,
    },
    "arthur conan": {
        "a study in scarlet": 3,
        "the sign of four": 2,
        "sherlock holmes the complete novels and stories volume i": 3,
        "the memoirs of sherlock holmes": 2,
        "the return of sherlock holmes": 2,
        "the complete sherlock holmes volume ii": 3,
        "the valley of fear": 2,
        "the casebook of sherlock holmes": 2,
        "a scandal in bohemia the adventures of sherlock holmes": 2,
        "the adventures of sherlock holmes": 3,
    },
    "markus": {
        "the dwarves": 1,
        "the war of dwarves": 1,
        "the revenge of the dwarves": 1,
        "i am the messenger": 3,
        "the book thief": 3,
        "getting the girl": 2,
        "fighting ruben wolfe": 2,
        "underdog": 2,
        "underdogs": 2,
    },
    "mockingbird": {
        "mockingbird": 2,
        "mockingbird songs": 1,
        "to kill a mockingbird": 3,
        "harper lee s to kill a mockingbird": 2,
    },
    "alchemist": {
        "alchemist": 1,
        "the alchemist": 3,
        "the neutronium alchemist": 1,
        "the blood alchemist": 1,
        "the alchemyst": 2,
        "fullmetal alchemist vol 1": 1,
        "the alchemist s secret": 1,
        "fullmetal alchemist vol 2": 1,
        "fullmetal alchemist vol 3": 1,
        "fullmetal alchemist vol 5": 1,
    },
    "fahrenheit": {
        "fahrenheit 451": 3,
        "the fahrenheit twins": 1,
        "a pleasure to burn fahrenheit 451 stories": 2,
    },
    "prejudice": {
        "pies prejudice": 0,
        "prom prejudice": 1,
        "pride and prejudice": 3,
        "pies and prejudice": 0,
        "penguin pain and prejudice": 0,
        "pride and prejudice and zombies": 2,
        "pride and prejudice ii the sequel": 1,
        "pride and prejudice retold in limericks": 0,
        "eligible a modern retelling of pride and prejudice": 2,
        "pride and prejudice and zombies the graphic novel": 1,
    },
    "narnia": {
        "the chronicles of narnia": 3,
        "the selected poetry of rainer maria rilke": 0,
        "my ntonia": 0,
        "marcia schuyler": 0,
        "diary of saint marua faustina kowalska divine mercy in my soul": 0,
        "sheet music the chronicles of narnia prince caspian": 2,
        "when marnie was there": 0,
        "fudge a mania": 0,
        "the lion the witch and the wardrobe chronicles of narnia 1 hiawyn oram": 3,
    },
    "harr potter": {
        "harry potter collection": 3,
        "the harry potter trilogy": 3,
        "harry potter film wizardry": 2,
        "harry potter the prequel": 2,
        "harry potter boxed set: books 1-5": 3,
        "harry potter series box set": 3,
        "the end of harry potter": 1,
        "harry hill's tv burp bookl": 0,
        "hard time": 0,
        "the tale of peter rabbit": 0,
    },
    "twiligt": {
        "twilight": 3,
        "twilight eyes": 1,
        "twilight watch": 1,
        "twilight falling": 1,
        "twilight phantasies": 1,
        "twilight illusions": 1,
        "twilight sleep": 1,
        "the twilight collection": 3,
        "the twilight saga": 3,
        "crossroads of twilight": 1,
    },
    "stpehenie meyer": {
        "female force bestsellers stephenie meyer": 2,
        "stephanie alexander s kitchen garden companion": 0,
        "stephanie lisa tara s turtle book ": 0,
        "the host": 2,
        "the twilight collection": 3,
        "the short second life of bree tanner": 2,
        "eclipse": 3,
        "breaking dawn": 3,
        "new moon": 3,
        "the twilight saga": 3,
    },
    "george orwel": {
        "george orwell omnibus the complete novels animal farm burmese days a clergyman's daughter coming up for air keep the aspidistra flying and nineteen eighty four": 3,
        "down and out in paris and london": 2,
        "animal farm 1984": 3,
        "the autobiography of george muller": 0,
        "homage to catalonia": 2,
        "keep the aspidistra flying": 2,
        "burmese days": 2,
        "coming up for air": 2,
        "animal farm": 3,
        "1984": 3,
    },
    "potter hogwarts": {
        "the hogwarts library": 3,
        "howards end": 0,
        "hogwarts an incomplete and unreliable guide": 3,
        "the tale of peter rabbit": 0,
        "we die alone a wwii epic of escape and endurance": 0,
        "the sledge patrol a wwii epic of escape survival and victory": 0,
        "harry potter schoolbooks box set two classic books from the library of hogwarts school of witchcraft and wizardry": 3,
        "short stories from hogwarts of heroism hardship and dangerous hobbies": 3,
        "short stories from hogwarts of power politics and pesky poltergeists": 3,
        "conversations with peter rosei": 0,
    },
    "mockingbird atticus": {
        "mockingbird": 2,
        "mockingbird songs": 1,
        "atticus": 2,
        "love her wild": 1,
        "to kill a mockingbird": 3,
        "teapots and attics don t let go teapots and attics": 0,
        "atticus claw breaks the law": 1,
        "harper lee s to kill a mockingbird": 2,
        "infantry attacks": 0,
        "teapots and attics i ll never let go": 0,
    },
    "alchemist coelho": {
        "the alchemist": 3,
        "alchemist": 1,
        "the neutronium alchemist": 1,
        "the blood alchemist": 1,
        "the alchemyst": 2,
        "fullmetal alchemist vol 1": 1,
        "the alchemist s secret": 1,
        "fullmetal alchemist vol 4": 1,
        "fullmetal alchemist vol 3": 1,
        "fullmetal alchemist vol 5": 1,
    },
}


def get_run_from_backend(query):
    resp = requests.get(
        f"{API_BASE}/compare_all_methods",
        params={"query": query}
    )
    return resp.json()


def build_run(method_results):
    run_completion = {}
    run_bm25       = {}
    run_sbert      = {}
    run_hybrid     = {}

    for query, data in method_results.items():
        # Completion — pakai posisi terbalik karena tidak ada skor numerik
        run_completion[query] = {}
        for i, item in enumerate(data["completion"]):
            run_completion[query][item["title"].lower()] = float(10 - i)

        # BM25
        run_bm25[query] = {}
        for item in data["bm25"]:
            run_bm25[query][item["title"].lower()] = float(item.get("score_bm25", 0.0))

        # SBERT
        run_sbert[query] = {}
        for item in data["sbert"]:
            run_sbert[query][item["title"].lower()] = float(item.get("score_sbert", 0.0))

        # Hybrid
        run_hybrid[query] = {}
        for item in data["hybrid"]:
            run_hybrid[query][item["title"].lower()] = float(item.get("score_hybrid", 0.0))

    return run_completion, run_bm25, run_sbert, run_hybrid



def hitung_ndcg(qrels, run, label):
    # Lowercase semua key di qrels supaya match dengan run
    qrels_lower = {
        q: {doc.lower(): score for doc, score in docs.items()}
        for q, docs in qrels.items()
        if q in run  # hanya query yang ada di run
    }
    run_filtered = {q: run[q] for q in qrels_lower}

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_lower, ["ndcg_cut_10"])
    results   = evaluator.evaluate(run_filtered)

    print(f"\n{'='*65}")
    print(f"  nDCG — {label}")
    print(f"{'='*65}")
    print(f"  {'Query':<25} {'nDCG@10':>10}")
    print(f"  {'-'*47}")

    total_10 = 0
    for query, metrics in results.items():
        s10 = metrics.get("ndcg_cut_10", 0)
        total_10 += s10
        print(f"  {query:<25} {s10:>10.4f}")

    n = len(results)
    print(f"  {'-'*47}")
    print(f"  {'Rata-rata':<25} {total_10/n:>10.4f}")
    print(f"{'='*65}")
    return total_10 / n


# ════════════════════════════════════════════════════════════════
# 4. MAIN
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Mengambil hasil dari backend...")

    method_results = {}
    for query in qrels.keys():
        print(f"  -> query: '{query}'")
        method_results[query] = get_run_from_backend(query)

    run_completion, run_bm25, run_sbert, run_hybrid = build_run(method_results)

    results_summary = {}
    results_summary["Completion Suggester"] = hitung_ndcg(qrels, run_completion, "Completion Suggester")
    results_summary["Re-ranking BM25"]      = hitung_ndcg(qrels, run_bm25,       "Re-ranking BM25")
    results_summary["Re-ranking SBERT"]     = hitung_ndcg(qrels, run_sbert,      "Re-ranking SBERT")
    results_summary["Re-ranking Hybrid"]    = hitung_ndcg(qrels, run_hybrid,     "Re-ranking Hybrid")

    # Ringkasan akhir
    print(f"\n{'='*65}")
    print("  PERBANDINGAN SEMUA METODE")
    print(f"{'='*65}")
    print(f"  {'Metode':<25}{'nDCG@10':>10}")
    print(f"  {'-'*47}")
    for metode, (s10) in results_summary.items():
        print(f"  {metode:<25} {s10:>10.4f}")
    print(f"{'='*65}")