import token

import pandas as pd
import random
import json
import math
import os
from datetime import datetime

BASE_DIR    = os.path.dirname(__file__)
CSV_FILE    = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "DATASET", "hasil_books2.csv"))
OUTPUT_JSON = "./v11test_queries_final.json"
LOG_FILE    = "./query_generation_qac.log"

# bagi adil, tpi perlu dipikirin lagi
PROPORSI = {
    "MODE1_PREFIX_TITLE"     : 0.25,    # Mode 1, LR (title)
    "MODE1_PREFIX_AUTHOR"    : 0.25,    # Mode 1, LR (author)
    "MODE2_MULTI_PREFIX"          : 0.25,   #Mode 2, LR
    "MODE4_TYPO"             : 0.25,   # Mode 4, GEN
}

# QWERTY ADJACENCY MAP  buat mode 4, jd si typonya berdasarkan karakter yg disampingnya
QWERTY = {
    'a': 'sqwz',  'b': 'vghn',  'c': 'xdfv',  'd': 'srfce',
    'e': 'wrsdf', 'f': 'dcvgt', 'g': 'ftbhy', 'h': 'gynj',
    'i': 'ujko',  'j': 'hunkm', 'k': 'jilm',  'l': 'kop',
    'm': 'njk',   'n': 'bhjm',  'o': 'iklp',  'p': 'ol',
    'q': 'wa',    'r': 'edft',  's': 'awedxz','t': 'rfgy',
    'u': 'yhji',  'v': 'cfgb',  'w': 'qase',  'x': 'zsdc',
    'y': 'tghu',  'z': 'asx',
}

STOP_WORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to",
    "for", "by", "with", "as", "is", "it", "its", "be", "was",
    "are", "from", "this", "that", "not", "but", "if", "up", "i",
    "do", "so", "no", "my", "we", "he", "she", "they", "us",
}

# Distribusi condong ke Level 1 (typo ringan), konsisten dengan
# model Gaussian σ=0.19 di Krishnan et al. (2017) Section 5.
#
#   Level  δ   Jumlah karakter ditypo   Proporsi
#   ─────  ──  ──────────────────────   ────────
#     1    1   1 karakter               70%
#     2    2   2 karakter               25%
#     3    3   3 karakter                5%
TYPO_LEVEL_PROBS = [
    (1, 0.70),   # Level 1 (δ=1) : 1 typo — 70%
    (2, 0.25),   # Level 2 (δ=2) : 2 typo — 25%
    (3, 0.05),   # Level 3 (δ=3) : 3 typo —  5%
]

# pembagian sample  pake rumus slovin
def slovin(N: int, e: float = 0.10) -> int:
    # N : int   — jumlah populasi (total records dataset)
    # e : float — margin error (default 0.10 = 10%)
    return math.ceil(N / (1 + N * (e ** 2)))

    
# Pilih jumlah karakter yang akan ditypo (1, 2, atau 3)
def _pick_typo_level() -> int:
    r     = random.random()
    cumul = 0.0
    for level, prob in TYPO_LEVEL_PROBS:
        cumul += prob
        if r < cumul:
            return level
    return 1

# substitusi satu karakter dengan tetangga QWERTY-nya.
def _substitute_char(ch: str) -> str:
    if ch in QWERTY:
        return random.choice(QWERTY[ch])
    return ch

# mode 1 prefix title dan author pake model LR dari kiri ke kanan tapi bukan karakter tapiper kata
def generate_mode1_prefix_title(title: str) -> str:
    # Strategi:
    # - Judul 1 kata  : ambil 3–5 karakter pertama (minimum 3 char,
    #                   sesuai batas bawah Bast & Weber dalam paper).
    # - Judul multi-kata: ambil 30–50% kata dari awal, minimal 2 kata,
    #                   tidak mengambil seluruh judul (harus ada sisa
    #                   ≥1 kata yang belum diketik, agar benar-benar
    #                   merepresentasikan query parsial Mode 1)
    words = title.strip().split()

    if len(words) == 1:
        cut = random.randint(3, min(5, len(words[0])))
        return words[0][:cut]

    # 30–50% dari total kata, minimal 2, maksimal len-1
    low  = max(2, math.ceil(len(words) * 0.30))
    high = max(2, math.floor(len(words) * 0.50))
    high = min(high, len(words) - 1)
    if low > high:
        low = high

    n_take = random.randint(low, high)
    if words[0].lower() in STOP_WORDS:
        n_take = max(n_take, min(3, len(words) - 1))

    return " ".join(words[:n_take])

def generate_mode1_prefix_author(author: str) -> str:
    # Strategi:
    # - Tolak author dengan token inisial tunggal (< 3 karakter)
    #   karena "j m" atau "j k" terlalu ambigu untuk menemukan
    #   satu pengarang secara spesifik.
    # - Ambil 2 token pertama nama (nama depan + nama belakang)
    #   agar prefix cukup spesifik.
    # - Tambahkan variasi panjang prefix karakter pada token terakhir
    #   untuk mensimulasikan pengguna yang belum selesai mengetik
    #   (misalnya: "andrea z" bukan hanya "andrea zuvich").
    words = author.strip().split()

    if any(len(w) < 3 for w in words):
        return ""

    # Ambil maks 2 token nama
    n_take = min(2, len(words))
    chosen = words[:n_take]

    # Potong token terakhir sebesar 30–70% panjangnya
    # agar mensimulasikan pengguna yang baru mulai mengetik
    last   = chosen[-1]
    cut    = random.randint(
        max(3, math.ceil(len(last) * 0.30)),
        max(3, math.ceil(len(last) * 0.70))
    )
    chosen[-1] = last[:cut]

    return " ".join(chosen)

# def generate_MODE2_MULTI_PREFIX(title: str) -> str:
#     """
#     Model LR untuk Mode 3 — PATTERN MATCH.
#     Mengambil substring dari tengah/akhir judul, termasuk
#     substring internal kata (bukan hanya prefix kata).

#     Aturan:
#     - Pool dijamin >= 4 kata dari generate_queries.
#     - Mulai dari index >= 1 (bukan kata pertama).
#     - Untuk setiap kata yang dipilih: 30% kemungkinan diambil
#       sebagai substring internal (misalnya "learning" → "earn"),
#       70% diambil sebagai prefix kata tersebut.
#     - Hindari menghasilkan query yang hanya stop word.
#     """
#     words = title.strip().split()

#     if len(words) < 3:
#         # Safeguard: kembalikan kata terakhir bukan stop word
#         for w in reversed(words):
#             if w.lower() not in STOP_WORDS and len(w) >= 3:
#                 return w
#         return words[-1]

#     # Mulai dari index 1 (bukan awal)
#     max_start = len(words) - 1
#     start = random.randint(1, max_start)
#     min_words = 3


#     # Tentukan jumlah kata yang diambil (1 atau 2)
#     n_words = 1
#     if start < len(words) - 1:
#         n_words = random.randint(1, 2)

#     chunk = words[start:start + n_words]

#     # Hindari chunk yang semua stop word
#     if all(w.lower() in STOP_WORDS for w in chunk):
#         # Cari 2 kata bukan stop word dari tengah/akhir
#         for s in range(1, len(words)):
#             c = [w for w in words[s:s + 2] if w.lower() not in STOP_WORDS]
#             if c:
#                 chunk = c
#                 break

#     # Transformasi token: ada probabilitas jadi substring internal
#     result_tokens = []
#     for token in chunk:
#         if len(token) >= 4 and random.random() < 0.30:
#             # Substring internal: mulai dari posisi 1 s/d len-2
#             max_inner_start = len(token) - 2
#             inner_start = random.randint(1, max_inner_start)
#             inner_len   = random.randint(
#                 max(2, math.ceil((len(token) - inner_start) * 0.50)),
#                 len(token) - inner_start
#             )
#             result_tokens.append(token[inner_start:inner_start + inner_len])
#         else:
#             # Prefix biasa dari token tersebut (bukan prefix judul)
#             cut = random.randint(
#                 max(2, math.ceil(len(token) * 0.40)),
#                 max(2, math.ceil(len(token) * 0.80))
#             )
#             result_tokens.append(token[:cut])

#     query = " ".join(result_tokens).strip()
#     return query if query else words[-1]


# mode 2 pattern match pake model LR juga tapi boleh ambil token dari posisi tengah atau akhir judul, termasuk substring internal kata (bukan hanya prefix)
def make_prefix(token: str) -> str:
    min_len = max(5, math.ceil(len(token) * 0.50))
    max_len = max(min_len, math.ceil(len(token) * 0.80))
    max_len = min(max_len, len(token) - 1)

    if min_len > max_len:
        min_len = max_len   
        
    cut = random.randint(min_len, max_len)
    return token[:cut]
        
def generate_MODE2_MULTI_PREFIX(title: str) -> str:
    # Aturan:
    # - Token diambil dari posisi index >= 1 di judul (non-awal).
    # - Hanya token NON-STOP WORD dengan len >= 5.
    # - 85% kasus: prefix token (potong 50–80% panjang token).
    # - 15% kasus: buang 1–2 char pertama token (simulasi suffix ringan).
    # - Panjang minimum substring: max(5, ceil(50% × len(token))).
    # - Diutamakan 2 token; urutan token output diacak.
    words = title.strip().split()
 
    # Pool: token NON-STOP WORD dari posisi index >= 1 (bukan kata pertama)
    # dengan panjang >= 5 agar bisa menghasilkan substring bermakna
    non_stop_nonfirst = [
        w for i, w in enumerate(words)
        if i >= 1
        and w.lower() not in STOP_WORDS
        and len(w) >= 5
    ]
 
    # Fallback: jika pool non-awal kosong, izinkan semua posisi
    if not non_stop_nonfirst:
        non_stop_nonfirst = [
            w for w in words
            if w.lower() not in STOP_WORDS and len(w) >= 5
        ]
 
    #gagal
    if not non_stop_nonfirst:
        return ""
 
    # Urutkan terpanjang → terpendek; bobot seleksi = panjang token
    pool_sorted = sorted(non_stop_nonfirst, key=len, reverse=True)
    weights     = [len(w) for w in pool_sorted]
 
    def weighted_sample_no_replace(population, wts, k):
        chosen = []
        pop = list(zip(wts, population))
        for _ in range(k):
            if not pop:
                break
            total  = sum(w for w, _ in pop)
            r      = random.uniform(0, total)
            cumsum = 0
            for idx, (w, item) in enumerate(pop):
                cumsum += w
                if cumsum >= r:
                    chosen.append(item)
                    pop.pop(idx)
                    break
        return chosen
 
    n_pick = min(2, len(pool_sorted))
    picked = weighted_sample_no_replace(pool_sorted, weights, n_pick)
 
    result_tokens = []
    for token in picked:
        sub = make_prefix(token)
        if sub:
            result_tokens.append(sub)
 
    # Acak urutan token output (Mode 3 bebas posisi, tidak harus urut judul)
    random.shuffle(result_tokens)
 
    query = " ".join(result_tokens).strip()
    return query

# mode 4 relaxed pattern match pake model GEN, simulasi kesalahan ketik berbasis keyboard QWERTY adjacency (Gaussian distribution, σ=0.19), dengan distribusi level typo 70/25/5 untuk 1/2/3 karakter ditypo
#    Mekanisme typo:
#      1. Tentukan jumlah typo (δ) berdasarkan distribusi level.
    # 2. Kumpulkan semua posisi karakter alfabet yang bisa ditypo
    #    dari seluruh kata BUKAN stop word.
    # 3. Pilih δ posisi yang BERBEDA secara acak (tanpa pengulangan
    #    pada posisi yang sama — sesuai poin 6 spesifikasi).
    # 4. Terapkan substitusi QWERTY pada posisi-posisi tersebut.
    # 5. Batasi δ ke jumlah karakter alfabet yang tersedia jika
    #    judul terlalu pendek (sesuai poin 5 spesifikasi).
#
# Merepresentasikan parameter toleransi error δ pada Eq. 7 paper.

def generate_mode4_typo(text: str) -> tuple[str, int]:
    words = text.strip().split()

    # Hanya dari kata bukan stop word dengan panjang >= 3
    eligible_positions = []
    for wi, word in enumerate(words):
        if word.lower() in STOP_WORDS or len(word) < 3:
            continue
        for ci, ch in enumerate(word):
            if ch.isalpha() and ch in QWERTY:
                eligible_positions.append((wi, ci))

    if not eligible_positions:
        longest_wi = max(range(len(words)), key=lambda i: len(words[i]))
        for ci, ch in enumerate(words[longest_wi]):
            if ch.isalpha() and ch in QWERTY:
                eligible_positions.append((longest_wi, ci))

    if not eligible_positions:
        return text, 0

    # tentukan level, batasi ke jumlah posisi tersedia
    target_delta = _pick_typo_level()
    actual_delta = min(target_delta, len(eligible_positions))

    # memilih posisi yang BERBEDA (tanpa duplikat) secara acak
    chosen_positions = set()
    sampled = random.sample(eligible_positions, actual_delta)
    for pos in sampled:
        chosen_positions.add(pos)

    # substitusi pada posisi yg kepilih
    result_words = [list(w) for w in words]
    applied = 0
    for (wi, ci) in chosen_positions:
        original_ch = result_words[wi][ci]
        new_ch      = _substitute_char(original_ch)
        result_words[wi][ci] = new_ch
        if new_ch != original_ch:
            applied += 1

    query_text = " ".join("".join(chars) for chars in result_words)
    return query_text, applied

# generate query berdasrakan mode 1,3,4
def generate_queries(df: pd.DataFrame, n_total: int, seed: int = 42) -> list[dict]:
    random.seed(seed)
    books = df.to_dict('records')

    n_pt  = math.ceil(n_total * PROPORSI["MODE1_PREFIX_TITLE"])
    n_pa  = math.ceil(n_total * PROPORSI["MODE1_PREFIX_AUTHOR"])
    n_m3  = math.ceil(n_total * PROPORSI["MODE2_MULTI_PREFIX"])
    n_m4  = n_total - n_pt - n_pa - n_m3  

    print(f"\n{'='*62}")
    print(f"  MODE1_PREFIX_TITLE      (Mode 1, LR ) : {n_pt:3d} query ")
    print(f"  MODE1_PREFIX_AUTHOR     (Mode 1, LR ) : {n_pa:3d} query ")
    print(f"  MODE2_MULTI_PREFIX           (Mode 3, LR ) : {n_m3:3d} query ")
    print(f"  MODE4_TYPO              (Mode 4, GEN) : {n_m4:3d} query ")
    print(f"{'='*62}\n")

    queries    = []
    used_texts = set()   # cegah duplikat

    def try_add(query_text: str, mode_qac: str,
                model_gen: str, book: dict,
                typo_level: int | None = None) -> bool:
        
        qt = query_text.strip().lower()
        if not qt or qt in used_texts:
            return False
        used_texts.add(qt)
        queries.append({
            "query_id"  : f"Q{len(queries)+1:04d}",
            "query_text": query_text.strip(),
            "mode_qac"  : mode_qac,
            "model_gen" : model_gen,
            "typo_level": typo_level,
            "title"     : book["title"],
            "author"    : book.get("author", "unknown"),
        })
        return True

    # mode 1 title
    pool = [b for b in books
            if isinstance(b["title"], str)
            and len(b["title"].split()) >= 3]
    random.shuffle(pool)
    count = 0
    for book in pool:
        if count >= n_pt:
            break
        q = generate_mode1_prefix_title(book["title"])
        if try_add(q, "MODE1_PREFIX_TITLE", "LR", book):
            count += 1
    print(f"  OK MODE1_PREFIX_TITLE      : {count} query terbentuk")

    # author
    pool = [b for b in books
            if isinstance(b.get("author"), str)
            and len(b["author"].strip().split()) >= 2
            and all(len(w) >= 3 for w in b["author"].strip().split())]
    random.shuffle(pool)
    count = 0
    for book in pool:
        if count >= n_pa:
            break
        q = generate_mode1_prefix_author(book["author"])
        if not q:
            continue
        if try_add(q, "MODE1_PREFIX_AUTHOR", "LR", book):
            count += 1
    print(f"  OK MODE1_PREFIX_AUTHOR     : {count} query terbentuk")
    
    # mode 3
    def has_enough_long_tokens(title_str: str) -> bool:
        toks = [w for w in title_str.split()
                if w.lower() not in STOP_WORDS and len(w) >= 5]
        return len(toks) >= 2
 
    pool = [b for b in books
            if isinstance(b["title"], str)
            and len(b["title"].split()) >= 4
            and has_enough_long_tokens(b["title"])]
    random.shuffle(pool)
    count = 0
    for book in pool:
        if count >= n_m3:
            break
        q = generate_MODE2_MULTI_PREFIX(book["title"])
        if try_add(q, "MODE2_MULTI_PREFIX", "LR", book):
            count += 1
    print(f"  OK MODE2_MULTI_PREFIX           : {count} query terbentuk")

    # mode 4
    pool = [b for b in books
            if isinstance(b["title"], str)
            and len(b["title"].split()) >= 2]
    random.shuffle(pool)
    count        = 0
    level_counts = {1: 0, 2: 0, 3: 0}   
    for book in pool:
        if count >= n_m4:
            break
        q, delta = generate_mode4_typo(book["title"])
        
        if delta > 0 and q.lower() != book["title"].lower():
            if try_add(q, "MODE4_TYPO", "GEN", book, typo_level=delta):
                count += 1
                level_counts[min(delta, 3)] += 1
    print(f"  OK MODE4_TYPO              : {count} query terbentuk")
    print(f"     Level 1 (delta=1): {level_counts[1]} query")
    print(f"     Level 2 (delta=2): {level_counts[2]} query")
    print(f"     Level 3 (delta=3): {level_counts[3]} query")

    return queries


def log(msg: str):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_json(data: list, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log(f"Tersimpan -> {path}  ({len(data)} queries)")


def print_sample(queries: list, n: int = 5):
    print(f"\n{'='*62}")
    print(f"  PREVIEW {n} QUERY PERTAMA")
    print(f"{'='*62}")
    for q in queries[:n]:
        print(f"\n  [{q['query_id']}] \"{q['query_text']}\"")
        print(f"       mode_qac   : {q['mode_qac']}")
        print(f"       model_gen  : {q['model_gen']}")
        lvl = q.get('typo_level')
        if lvl is not None:
            print(f"       typo_level : {lvl} (delta={lvl})")
        print(f"       Title      : {q['title']}")
        print(f"       Author     : {q['author']}")



def main():
    # Reset log
    with open(LOG_FILE, "w") as f:
        f.write("")

    
    log(f"Loading CSV: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE, usecols=["title", "author"])
    df = df.dropna(subset=["title"])
    df["title"]  = df["title"].str.lower().str.strip()
    df["author"] = df["author"].fillna("").str.lower().str.strip()
    df = df.drop_duplicates(subset=["title"])
    df = df.reset_index(drop=True)
    N  = len(df)
    log(f"Dataset: {N:,} records unik")


    e = 0.10   
    n = slovin(N, e)
    log(f"Rumus Slovin -> N={N:,}, e={e*100:.0f}% -> n={n}")
    print(f"\n  Perhitungan Slovin:")
    print(f"  n = {N} / (1 + {N} x {e}^2)")
    print(f"  n = {N} / (1 + {N * e**2:.2f})")
    print(f"  n = {N} / {1 + N * e**2:.2f}")
    print(f"  n = {N / (1 + N * e**2):.4f} -> dibulatkan ke atas = {n}")

    queries = generate_queries(df, n_total=n)

    save_json(queries, OUTPUT_JSON)
    print_sample(queries, n=5)

    log("=" * 62)
    log(f"DONE! {len(queries)} query uji tersimpan -> {OUTPUT_JSON}")
    log("=" * 62)


if __name__ == "__main__":
    main()