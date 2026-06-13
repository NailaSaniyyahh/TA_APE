# """
# queryUji.py
# ===========
# Pembentukan Query Uji untuk Eksperimen Search Suggestion

# Justifikasi:
# - Jumlah query  : Rumus Slovin (Nurjannah Syakrani & Naufal Athaya, JACOST 2025)
# - Kategori query: Taksonomi QAC Krishnan et al. (ADCS 2017)
#     * PREFIX_TITLE  → Mode 1 (Prefix Match),          Model LR
#     * PREFIX_AUTHOR → Mode 1 (Prefix Match),          Model LR
#     * PARTIAL       → Mode 3 (Pattern Match),         Model LR
#     * TYPO          → Mode 4 (Relaxed Pattern Match), Model GEN
# """

# import pandas as pd
# import random
# import json
# import math
# import os
# from datetime import datetime

# # ─────────────────────────────────────────────────────────────────
# # KONFIGURASI PATH  →  sesuaikan dengan struktur folder kamu
# # ─────────────────────────────────────────────────────────────────
# CSV_FILE    = "../hasil_books2.csv"          # ganti dengan path CSV kamu
# OUTPUT_JSON = "../test_queries_final.json"   # output query uji
# LOG_FILE    = "../query_generation.log"

# # ─────────────────────────────────────────────────────────────────
# # PROPORSI KATEGORI
# # ─────────────────────────────────────────────────────────────────
# PROPORSI = {
#     "PREFIX_TITLE" : 0.30,   # 30%
#     "PREFIX_AUTHOR": 0.20,   # 20%
#     "PARTIAL"      : 0.30,   # 30%
#     "TYPO"         : 0.20,   # 20%
# }

# # ─────────────────────────────────────────────────────────────────
# # QWERTY NEIGHBORS  (untuk simulasi typo Model GEN)
# # Sumber: Krishnan et al. (2017) Section 5
# # ─────────────────────────────────────────────────────────────────
# QWERTY = {
#     'a':'sqwz',  'b':'vghn',  'c':'xdfv',  'd':'srfce',
#     'e':'wrsdf', 'f':'dcvgt', 'g':'ftbhy', 'h':'gynj',
#     'i':'ujko',  'j':'hunkm', 'k':'jilm',  'l':'kop',
#     'm':'njk',   'n':'bhjm',  'o':'iklp',  'p':'ol',
#     'q':'wa',    'r':'edft',  's':'awedxz','t':'rfgy',
#     'u':'yhji',  'v':'cfgb',  'w':'qase',  'x':'zsdc',
#     'y':'tghu',  'z':'asx',
# }

# # =================================================================
# # 1. RUMUS SLOVIN
# #    Referensi: Nurjannah Syakrani & Naufal Athaya (2025)
# #               JACOST Vol.6 No.1 — Persamaan (4)
# #    n = N / (1 + N * e²)
# # =================================================================
# def slovin(N: int, e: float = 0.10) -> int:
#     """
#     Menghitung ukuran sampel menggunakan rumus Slovin.

#     Parameter
#     ----------
#     N : int   — jumlah populasi (total records dataset)
#     e : float — margin error (default 0.10 = 10%)

#     Return
#     ------
#     n : int   — jumlah query uji yang harus dibentuk
#     """
#     n = N / (1 + N * (e ** 2))
#     return math.ceil(n)


# # =================================================================
# # 2. MODEL LR — PREFIX  (Mode 1, Krishnan et al. 2017)
# #    Mengambil potongan dari bagian AWAL teks (kiri ke kanan).
# #    Merepresentasikan pengguna yang mengetik kata secara bertahap.
# # =================================================================
# def generate_prefix_LR(text: str) -> str:
#     """
#     Model LR dari Krishnan et al. (2017) Section 5, Tabel 4.
#     Append karakter dari kiri ke kanan → ambil prefix 30-50%
#     dari total kata.

#     Justifikasi panjang prefix:
#     - Paper menyebut minimum length = 3 karakter (Bast & Weber
#       dalam Krishnan et al. 2017 Section 5).
#     - 30-50% dipilih agar prefix tidak terlalu pendek (ambigu)
#       dan tidak terlalu panjang (terlalu mudah ditebak),
#       merepresentasikan pengguna yang baru mulai mengetik.

#     Contoh:
#       "the hunger games"    (3 kata) -> ambil 1 kata  -> "the"
#       "pride and prejudice" (3 kata) -> ambil 1-2 kata -> "pride and"
#       "gone with the wind"  (4 kata) -> ambil 1-2 kata -> "gone with"
#     """
#     words = text.strip().split()

#     if len(words) == 1:
#         # Untuk judul 1 kata: ambil 3-5 karakter pertama
#         cut = random.randint(3, min(5, len(words[0])))
#         return words[0][:cut]

#     # Hitung rentang 30-50% dari total kata, minimum 1
#     low  = max(1, math.ceil(len(words) * 0.30))
#     high = max(1, math.floor(len(words) * 0.50))

#     # Pastikan high tidak melebihi panjang - 1 (tidak ambil semua)
#     high = min(high, len(words) - 1)

#     # Jika low > high karena pembulatan, samakan
#     if low > high:
#         low = high

#     n_take = random.randint(low, high)
#     return " ".join(words[:n_take])


# # =================================================================
# # 3. MODEL LR — PARTIAL  (Mode 3, Krishnan et al. 2017)
# #    Mengambil kata dari posisi TENGAH atau AKHIR judul.
# #    Merepresentasikan pengguna yang hanya ingat sebagian kata.
# # =================================================================
# def generate_partial_LR(text: str) -> str:
#     """
#     Model LR untuk Mode 3 (Pattern Match).
#     Kata diambil bukan dari posisi awal (index > 0),
#     sehingga tidak bisa ditangkap prefix match biasa.

#     Aturan:
#     - ≤2 kata : kembalikan kata terakhir
#     - ≥3 kata : pilih 1–2 kata mulai dari index ≥ 1
#     """
#     words = text.strip().split()

#     if len(words) <= 2:
#         return words[-1]

#     # Mulai dari index 1 (bukan awal)
#     start = random.randint(1, len(words) - 1)
#     end   = min(start + random.randint(0, 1), len(words) - 1)
#     return " ".join(words[start:end + 1])


# # =================================================================
# # 4. MODEL GEN — TYPO  (Mode 4, Krishnan et al. 2017)
# #    Simulasi kesalahan ketik berdasarkan kedekatan karakter
# #    pada layout QWERTY (distribusi Gaussian, σ = 0.19).
# #    P_append = 0.80, P_delete = 0.04 (Section 5)
# # =================================================================
# def generate_typo_GEN(text: str, typo_prob: float = 0.25) -> str:
#     """
#     Model GEN dari Krishnan et al. (2017) Section 5.
#     Mensimulasikan typo ringan pada 1–2 kata dalam query.

#     typo_prob = probabilitas tiap karakter mengalami typo.
#     Nilai 0.25 menghasilkan rata-rata 1–2 typo per query
#     (masih menyerupai kata asli, sesuai batasan TA).
#     """
#     words = text.strip().split()

#     n_typo_words  = random.randint(1, min(2, len(words)))
#     idx_to_typo   = set(random.sample(range(len(words)), n_typo_words))

#     result = []
#     for i, word in enumerate(words):
#         if i in idx_to_typo:
#             new_chars = []
#             for ch in word:
#                 if ch.isalpha() and ch in QWERTY and random.random() < typo_prob:
#                     new_chars.append(random.choice(QWERTY[ch]))
#                 else:
#                     new_chars.append(ch)
#             result.append("".join(new_chars))
#         else:
#             result.append(word)

#     return " ".join(result)


# def generate_queries(df: pd.DataFrame, n_total: int,
#                      seed: int = 42) -> list[dict]:
#     """
#     Membentuk query uji sesuai jumlah Slovin dan 4 kategori.

#     Parameter
#     ----------
#     df      : DataFrame dengan kolom 'title' dan 'author'
#     n_total : jumlah total query (hasil Slovin)
#     seed    : random seed untuk reprodusibilitas
#     """
#     random.seed(seed)
#     books = df.to_dict('records')

#     n_pt = math.ceil(n_total * PROPORSI["PREFIX_TITLE"])
#     n_pa = math.ceil(n_total * PROPORSI["PREFIX_AUTHOR"])
#     n_par= math.ceil(n_total * PROPORSI["PARTIAL"])
#     n_ty = n_total - n_pt - n_pa - n_par   

#     print(f"\n{'='*55}")
#     print(f"  DISTRIBUSI QUERY (Total Slovin = {n_total})")
#     print(f"{'='*55}")
#     print(f"  PREFIX_TITLE   (Mode 1, LR ) : {n_pt:3d} query (30%)")
#     print(f"  PREFIX_AUTHOR  (Mode 1, LR ) : {n_pa:3d} query (20%)")
#     print(f"  PARTIAL        (Mode 3, LR ) : {n_par:3d} query (30%)")
#     print(f"  TYPO           (Mode 4, GEN) : {n_ty:3d} query (20%)")
#     print(f"{'='*55}\n")

#     queries    = []
#     used_texts = set() 

#     def try_add(query_text, qtype, book, difficulty, mode_ref) -> bool:
#         """Tambahkan query jika belum duplikat dan tidak kosong."""
#         qt = query_text.strip().lower()
#         if not qt or qt in used_texts:
#             return False
#         used_texts.add(qt)
#         queries.append({
#             "query_id"   : f"Q{len(queries)+1:03d}",
#             "query_text" : query_text.strip(),
#             "query_type" : qtype,
#             "title"      : book["title"],
#             "author"     : book.get("author", "unknown"),
#         })
#         return True

#     pool  = [b for b in books
#              if isinstance(b["title"], str)
#              and len(b["title"].split()) >= 2]
#     random.shuffle(pool)
#     count = 0
#     for book in pool:
#         if count >= n_pt:
#             break
#         q = generate_prefix_LR(book["title"])
#         if try_add(q, "PREFIX_TITLE", book, "EASY",
#                    "Mode 1 – Prefix Match (Krishnan et al., 2017), Model LR"):
#             count += 1
#     print(f"  ✓ PREFIX_TITLE   : {count} query terbentuk")

#     pool  = [b for b in books
#              if isinstance(b.get("author"), str)
#              and b["author"].strip()]
#     random.shuffle(pool)
#     count = 0
#     for book in pool:
#         if count >= n_pa:
#             break
#         q = generate_prefix_LR(book["author"])
#         if try_add(q, "PREFIX_AUTHOR", book, "EASY",
#                    "Mode 1 – Prefix Match (Krishnan et al., 2017), Model LR"):
#             count += 1
#     print(f"  ✓ PREFIX_AUTHOR  : {count} query terbentuk")

#     pool  = [b for b in books
#              if isinstance(b["title"], str)
#              and len(b["title"].split()) >= 3]
#     random.shuffle(pool)
#     count = 0
#     for book in pool:
#         if count >= n_par:
#             break
#         q = generate_partial_LR(book["title"])
#         if try_add(q, "PARTIAL", book, "MEDIUM",
#                    "Mode 3 – Pattern Match (Krishnan et al., 2017), Model LR"):
#             count += 1
#     print(f"  ✓ PARTIAL        : {count} query terbentuk")

#     pool  = [b for b in books
#              if isinstance(b["title"], str)
#              and len(b["title"].split()) >= 2]
#     random.shuffle(pool)
#     count = 0
#     for book in pool:
#         if count >= n_ty:
#             break
#         q = generate_typo_GEN(book["title"])
        
#         if q.lower() != book["title"].lower():
#             if try_add(q, "TYPO", book, "HARD",
#                        "Mode 4 – Relaxed Pattern Match (Krishnan et al., 2017), Model GEN"):
#                 count += 1
#     print(f"  ✓ TYPO           : {count} query terbentuk")

#     return queries

# def log(msg: str):
#     ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     line = f"[{ts}] {msg}"
#     print(line)
#     with open(LOG_FILE, "a", encoding="utf-8") as f:
#         f.write(line + "\n")

# def save_json(data: list, path: str):
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=2, ensure_ascii=False)
#     log(f"✓ Tersimpan → {path}  ({len(data)} queries)")

# def print_sample(queries: list, n: int = 5):
#     print(f"\n{'='*55}")
#     print(f"  PREVIEW {n} QUERY PERTAMA")
#     print(f"{'='*55}")
#     for q in queries[:n]:
#         print(f"\n  [{q['query_id']}] \"{q['query_text']}\"")
#         print(f"       Type  : {q['query_type']}")
#         print(f"       Title : {q['title']}")
#         print(f"       Author: {q['author']}")

# def print_statistics(queries: list):
#     from collections import Counter
#     types = Counter(q["query_type"] for q in queries)

#     print(f"\n{'='*55}")
#     print(f"  STATISTIK QUERY UJI")
#     print(f"{'='*55}")
#     print(f"  Total : {len(queries)} queries\n")
#     print("  TIPE:")
#     for t in ["PREFIX_TITLE","PREFIX_AUTHOR","PARTIAL","TYPO"]:
#         c   = types[t]
#         pct = c / len(queries) * 100
#         bar = "█" * int(pct / 5)
#         print(f"    {t:16} {c:3}  ({pct:5.1f}%)  {bar}")


# def main():
#     with open(LOG_FILE, "w") as f:
#         f.write("")

#     log("=" * 55)
#     log("START: Pembentukan Query Uji")
#     log("=" * 55)

#     log(f"Loading CSV: {CSV_FILE}")
#     df = pd.read_csv(CSV_FILE, usecols=["title", "author"])
#     df = df.dropna(subset=["title"])
#     df["title"]  = df["title"].str.lower().str.strip()
#     df["author"] = df["author"].str.lower().str.strip()
#     df = df.drop_duplicates(subset=["title"])
#     df = df.reset_index(drop=True)
#     N  = len(df)
#     log(f"✓ Dataset: {N:,} records unik")

#     e = 0.10  
#     n = slovin(N, e)
#     log(f"✓ Rumus Slovin → N={N:,}, e={e*100:.0f}% → n={n}")
#     print(f"\n  Perhitungan Slovin:")
#     print(f"  n = {N} / (1 + {N} × {e}²)")
#     print(f"  n = {N} / (1 + {N * e**2:.2f})")
#     print(f"  n = {N} / {1 + N * e**2:.2f}")
#     print(f"  n = {N / (1 + N * e**2):.4f} → dibulatkan ke atas = {n}")

#     queries = generate_queries(df, n_total=n)

#     save_json(queries, OUTPUT_JSON)
#     print_statistics(queries)
#     print_sample(queries, n=5)

#     log("=" * 55)
#     log(f"DONE! {len(queries)} query uji tersimpan → {OUTPUT_JSON}")
#     log("=" * 55)


# if __name__ == "__main__":
#     main()

"""
queryUji.py
===========
Pembentukan Query Uji untuk Eksperimen Search Suggestion

Justifikasi:
- Jumlah query  : Rumus Slovin (Nurjannah Syakrani & Naufal Athaya, JACOST 2025)
- Kategori query: Taksonomi QAC Krishnan et al. (ADCS 2017)
    * PREFIX_TITLE  → Mode 1 (Prefix Match),          Model LR
    * PREFIX_AUTHOR → Mode 1 (Prefix Match),          Model LR
    * PARTIAL       → Mode 3 (Pattern Match),         Model LR
    * TYPO          → Mode 4 (Relaxed Pattern Match), Model GEN
"""

# import pandas as pd
# import random
# import json
# import math
# import os
# from datetime import datetime

# # ─────────────────────────────────────────────────────────────────
# # KONFIGURASI PATH  →  sesuaikan dengan struktur folder kamu
# # ─────────────────────────────────────────────────────────────────
# CSV_FILE    = "../hasil_books2.csv"          # ganti dengan path CSV kamu
# OUTPUT_JSON = "../v2test_queries_final.json"   # output query uji
# LOG_FILE    = "../query_generation.log"

# # ─────────────────────────────────────────────────────────────────
# # PROPORSI KATEGORI
# # ─────────────────────────────────────────────────────────────────
# PROPORSI = {
#     "PREFIX_TITLE" : 0.30,   # 30%
#     "PREFIX_AUTHOR": 0.20,   # 20%
#     "PARTIAL"      : 0.30,   # 30%
#     "TYPO"         : 0.20,   # 20%
# }

# # ─────────────────────────────────────────────────────────────────
# # QWERTY NEIGHBORS  (untuk simulasi typo Model GEN)
# # Sumber: Krishnan et al. (2017) Section 5
# # ─────────────────────────────────────────────────────────────────
# QWERTY = {
#     'a':'sqwz',  'b':'vghn',  'c':'xdfv',  'd':'srfce',
#     'e':'wrsdf', 'f':'dcvgt', 'g':'ftbhy', 'h':'gynj',
#     'i':'ujko',  'j':'hunkm', 'k':'jilm',  'l':'kop',
#     'm':'njk',   'n':'bhjm',  'o':'iklp',  'p':'ol',
#     'q':'wa',    'r':'edft',  's':'awedxz','t':'rfgy',
#     'u':'yhji',  'v':'cfgb',  'w':'qase',  'x':'zsdc',
#     'y':'tghu',  'z':'asx',
# }

# # =================================================================
# # 1. RUMUS SLOVIN
# #    Referensi: Nurjannah Syakrani & Naufal Athaya (2025)
# #               JACOST Vol.6 No.1 — Persamaan (4)
# #    n = N / (1 + N * e²)
# # =================================================================
# def slovin(N: int, e: float = 0.10) -> int:
#     """
#     Menghitung ukuran sampel menggunakan rumus Slovin.

#     Parameter
#     ----------
#     N : int   — jumlah populasi (total records dataset)
#     e : float — margin error (default 0.10 = 10%)

#     Return
#     ------
#     n : int   — jumlah query uji yang harus dibentuk
#     """
#     n = N / (1 + N * (e ** 2))
#     return math.ceil(n)


# # =================================================================
# # 2. MODEL LR — PREFIX  (Mode 1, Krishnan et al. 2017)
# #    Mengambil potongan dari bagian AWAL teks (kiri ke kanan).
# #    Merepresentasikan pengguna yang mengetik kata secara bertahap.
# #
# #    FIX: minimum 2 kata agar query tidak ambigu (misal "the" saja
# #         bisa match ribuan buku → tidak representatif).
# #         for_author=True: minimal nama depan + belakang.
# # =================================================================
# def generate_prefix_LR(text: str, for_author: bool = False) -> str:
#     """
#     Model LR dari Krishnan et al. (2017) Section 5, Tabel 4.
#     Append karakter dari kiri ke kanan → ambil prefix 30-50%
#     dari total kata, MINIMUM 2 kata agar tidak ambigu.

#     Justifikasi panjang prefix:
#     - Paper menyebut minimum length = 3 karakter (Bast & Weber
#       dalam Krishnan et al. 2017 Section 5).
#     - Minimum 2 kata diterapkan agar query tidak terlalu ambigu.
#     - for_author=True: ambil minimal nama depan + nama belakang
#       agar query spesifik ke pengarang yang dimaksud.

#     Contoh PREFIX_TITLE:
#       "the hunger games"    (3 kata) -> "the hunger"
#       "pride and prejudice" (3 kata) -> "pride and"
#       "gone with the wind"  (4 kata) -> "gone with"

#     Contoh PREFIX_AUTHOR:
#       "andrea zuvich"  (2 kata) -> "andrea zuvich"
#       "j r r tolkien"  (4 kata) -> "j r r"
#     """
#     words = text.strip().split()

#     if len(words) == 1:
#         # 1 kata: ambil 3-5 karakter pertama
#         cut = random.randint(3, min(5, len(words[0])))
#         return words[0][:cut]

#     if for_author:
#         # Author: minimal 2 kata (nama depan + belakang), maks 75% kata
#         low    = min(2, len(words))
#         high   = min(len(words), max(2, math.floor(len(words) * 0.75)))
#         n_take = random.randint(low, high)
#         return " ".join(words[:n_take])

#     # Title: minimum 2 kata, rentang 30-50% dari total kata
#     low  = max(2, math.ceil(len(words) * 0.30))
#     high = max(2, math.floor(len(words) * 0.50))

#     # Pastikan high tidak melebihi panjang - 1 (tidak ambil semua)
#     high = min(high, len(words) - 1)

#     # Jika low > high karena pembulatan, samakan
#     if low > high:
#         low = high

#     n_take = random.randint(low, high)
#     return " ".join(words[:n_take])


# # =================================================================
# # 3. MODEL LR — PARTIAL  (Mode 3, Krishnan et al. 2017)
# #    Mengambil kata dari posisi TENGAH atau AKHIR judul.
# #    Merepresentasikan pengguna yang hanya ingat sebagian kata.
# # =================================================================
# def generate_partial_LR(text: str) -> str:
#     """
#     Model LR untuk Mode 3 (Pattern Match).
#     Kata diambil bukan dari posisi awal (index > 0),
#     sehingga tidak bisa ditangkap prefix match biasa.

#     Aturan:
#     - ≤2 kata : kembalikan kata terakhir
#     - ≥3 kata : pilih 1-2 kata mulai dari index >= 1
#     """
#     words = text.strip().split()

#     if len(words) <= 2:
#         return words[-1]

#     # Mulai dari index 1 (bukan awal)
#     start = random.randint(1, len(words) - 1)
#     end   = min(start + random.randint(0, 1), len(words) - 1)
#     return " ".join(words[start:end + 1])


# # =================================================================
# # 4. MODEL GEN — TYPO  (Mode 4, Krishnan et al. 2017)
# #    Simulasi kesalahan ketik berdasarkan kedekatan karakter
# #    pada layout QWERTY (distribusi Gaussian, σ = 0.19).
# #    P_append = 0.80, P_delete = 0.04 (Section 5)
# #
# #    FIX: batasi maksimal 2 karakter yang ditypo per query
# #         agar hasil typo masih mirip kata aslinya dan sistem
# #         masih bisa menemukannya (Relaxed Pattern Match).
# # =================================================================
# def generate_typo_GEN(text: str, typo_prob: float = 0.25,
#                       max_typo_chars: int = 2) -> str:
#     """
#     Model GEN dari Krishnan et al. (2017) Section 5.
#     Mensimulasikan typo ringan pada 1-2 kata dalam query,
#     dengan MAKSIMAL 2 karakter yang berubah per query.

#     typo_prob     = probabilitas tiap karakter mengalami typo.
#     max_typo_chars = batas atas jumlah karakter yang boleh ditypo
#                      per query (default 2), agar query masih
#                      menyerupai kata asli dan bisa ditemukan sistem.

#     Contoh yang diharapkan:
#       "the hunger games" -> "the hunfer games"  (1 typo)
#       "above world"      -> "abpve world"        (1 typo)
#     """
#     words = text.strip().split()

#     # Pilih 1-2 kata yang akan ditypo (bukan semua kata)
#     n_typo_words = random.randint(1, min(2, len(words)))
#     idx_to_typo  = set(random.sample(range(len(words)), n_typo_words))

#     result         = []
#     total_typo_count = 0   # hitung total karakter yang sudah ditypo

#     for i, word in enumerate(words):
#         if i in idx_to_typo and total_typo_count < max_typo_chars:
#             new_chars = []
#             for ch in word:
#                 # Hentikan typo jika sudah capai batas max
#                 if (total_typo_count < max_typo_chars
#                         and ch.isalpha()
#                         and ch in QWERTY
#                         and random.random() < typo_prob):
#                     new_chars.append(random.choice(QWERTY[ch]))
#                     total_typo_count += 1
#                 else:
#                     new_chars.append(ch)
#             result.append("".join(new_chars))
#         else:
#             result.append(word)

#     return " ".join(result)


# # =================================================================
# # 5. GENERATE QUERY UJI
# # =================================================================
# def generate_queries(df: pd.DataFrame, n_total: int,
#                      seed: int = 42) -> list[dict]:
#     """
#     Membentuk query uji sesuai jumlah Slovin dan 4 kategori.

#     Parameter
#     ----------
#     df      : DataFrame dengan kolom 'title' dan 'author'
#     n_total : jumlah total query (hasil Slovin)
#     seed    : random seed untuk reprodusibilitas
#     """
#     random.seed(seed)
#     books = df.to_dict('records')

#     # Hitung jumlah per kategori
#     n_pt  = math.ceil(n_total * PROPORSI["PREFIX_TITLE"])
#     n_pa  = math.ceil(n_total * PROPORSI["PREFIX_AUTHOR"])
#     n_par = math.ceil(n_total * PROPORSI["PARTIAL"])
#     n_ty  = n_total - n_pt - n_pa - n_par   # sisa ke TYPO

#     print(f"\n{'='*55}")
#     print(f"  DISTRIBUSI QUERY (Total Slovin = {n_total})")
#     print(f"{'='*55}")
#     print(f"  PREFIX_TITLE   (Mode 1, LR ) : {n_pt:3d} query (30%)")
#     print(f"  PREFIX_AUTHOR  (Mode 1, LR ) : {n_pa:3d} query (20%)")
#     print(f"  PARTIAL        (Mode 3, LR ) : {n_par:3d} query (30%)")
#     print(f"  TYPO           (Mode 4, GEN) : {n_ty:3d} query (20%)")
#     print(f"{'='*55}\n")

#     queries    = []
#     used_texts = set()   # cegah duplikat

#     def try_add(query_text, qtype, book, difficulty, mode_ref) -> bool:
#         """Tambahkan query jika belum duplikat dan tidak kosong."""
#         qt = query_text.strip().lower()
#         if not qt or qt in used_texts:
#             return False
#         used_texts.add(qt)
#         queries.append({
#             "query_id"   : f"Q{len(queries)+1:03d}",
#             "query_text" : query_text.strip(),
#             "query_type" : qtype,
#             "title"      : book["title"],
#             "author"     : book.get("author", "unknown"),
#         })
#         return True

#     # -- PREFIX TITLE (Mode 1, LR) ---------------------------------
#     # FIX: filter judul >= 2 kata, generate_prefix_LR minimum 2 kata
#     pool  = [b for b in books
#              if isinstance(b["title"], str)
#              and len(b["title"].split()) >= 2]
#     random.shuffle(pool)
#     count = 0
#     for book in pool:
#         if count >= n_pt:
#             break
#         q = generate_prefix_LR(book["title"], for_author=False)
#         if try_add(q, "PREFIX_TITLE", book, "EASY",
#                    "Mode 1 - Prefix Match (Krishnan et al., 2017), Model LR"):
#             count += 1
#     print(f"  OK PREFIX_TITLE   : {count} query terbentuk")

#     # -- PREFIX AUTHOR (Mode 1, LR) --------------------------------
#     # FIX: filter author >= 2 kata, pakai for_author=True agar
#     #      query minimal nama depan+belakang (lebih spesifik)
#     pool  = [b for b in books
#              if isinstance(b.get("author"), str)
#              and len(b["author"].strip().split()) >= 2]
#     random.shuffle(pool)
#     count = 0
#     for book in pool:
#         if count >= n_pa:
#             break
#         q = generate_prefix_LR(book["author"], for_author=True)
#         if try_add(q, "PREFIX_AUTHOR", book, "EASY",
#                    "Mode 1 - Prefix Match (Krishnan et al., 2017), Model LR"):
#             count += 1
#     print(f"  OK PREFIX_AUTHOR  : {count} query terbentuk")

#     # -- PARTIAL (Mode 3, LR) --------------------------------------
#     pool  = [b for b in books
#              if isinstance(b["title"], str)
#              and len(b["title"].split()) >= 3]
#     random.shuffle(pool)
#     count = 0
#     for book in pool:
#         if count >= n_par:
#             break
#         q = generate_partial_LR(book["title"])
#         if try_add(q, "PARTIAL", book, "MEDIUM",
#                    "Mode 3 - Pattern Match (Krishnan et al., 2017), Model LR"):
#             count += 1
#     print(f"  OK PARTIAL        : {count} query terbentuk")

#     # -- TYPO (Mode 4, GEN) ----------------------------------------
#     # FIX: max_typo_chars=2 agar tidak lebih dari 2 karakter berubah
#     pool  = [b for b in books
#              if isinstance(b["title"], str)
#              and len(b["title"].split()) >= 2]
#     random.shuffle(pool)
#     count = 0
#     for book in pool:
#         if count >= n_ty:
#             break
#         q = generate_typo_GEN(book["title"], typo_prob=0.25, max_typo_chars=2)
#         # Pastikan hasil typo berbeda dari aslinya
#         if q.lower() != book["title"].lower():
#             if try_add(q, "TYPO", book, "HARD",
#                        "Mode 4 - Relaxed Pattern Match (Krishnan et al., 2017), Model GEN"):
#                 count += 1
#     print(f"  OK TYPO           : {count} query terbentuk")

#     return queries


# # =================================================================
# # 6. LOGGING & SIMPAN
# # =================================================================
# def log(msg: str):
#     ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     line = f"[{ts}] {msg}"
#     print(line)
#     with open(LOG_FILE, "a", encoding="utf-8") as f:
#         f.write(line + "\n")

# def save_json(data: list, path: str):
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=2, ensure_ascii=False)
#     log(f"Tersimpan -> {path}  ({len(data)} queries)")

# def print_sample(queries: list, n: int = 5):
#     print(f"\n{'='*55}")
#     print(f"  PREVIEW {n} QUERY PERTAMA")
#     print(f"{'='*55}")
#     for q in queries[:n]:
#         print(f"\n  [{q['query_id']}] \"{q['query_text']}\"")
#         print(f"       Type  : {q['query_type']}")
#         print(f"       Title : {q['title']}")
#         print(f"       Author: {q['author']}")

# def print_statistics(queries: list):
#     from collections import Counter
#     types = Counter(q["query_type"] for q in queries)

#     print(f"\n{'='*55}")
#     print(f"  STATISTIK QUERY UJI")
#     print(f"{'='*55}")
#     print(f"  Total : {len(queries)} queries\n")
#     print("  TIPE:")
#     for t in ["PREFIX_TITLE","PREFIX_AUTHOR","PARTIAL","TYPO"]:
#         c   = types[t]
#         pct = c / len(queries) * 100
#         bar = "X" * int(pct / 5)
#         print(f"    {t:16} {c:3}  ({pct:5.1f}%)  {bar}")


# # =================================================================
# # MAIN
# # =================================================================
# def main():
#     # Reset log
#     with open(LOG_FILE, "w") as f:
#         f.write("")

#     log("=" * 55)
#     log("START: Pembentukan Query Uji")
#     log("=" * 55)

#     # -- Load CSV --------------------------------------------------
#     log(f"Loading CSV: {CSV_FILE}")
#     df = pd.read_csv(CSV_FILE, usecols=["title", "author"])
#     df = df.dropna(subset=["title"])
#     df["title"]  = df["title"].str.lower().str.strip()
#     df["author"] = df["author"].fillna("").str.lower().str.strip()
#     df = df.drop_duplicates(subset=["title"])
#     df = df.reset_index(drop=True)
#     N  = len(df)
#     log(f"Dataset: {N:,} records unik")

#     # -- Hitung n (Rumus Slovin) -----------------------------------
#     e = 0.10   # margin error 10%
#     n = slovin(N, e)
#     log(f"Rumus Slovin -> N={N:,}, e={e*100:.0f}% -> n={n}")
#     print(f"\n  Perhitungan Slovin:")
#     print(f"  n = {N} / (1 + {N} x {e}^2)")
#     print(f"  n = {N} / (1 + {N * e**2:.2f})")
#     print(f"  n = {N} / {1 + N * e**2:.2f}")
#     print(f"  n = {N / (1 + N * e**2):.4f} -> dibulatkan ke atas = {n}")

#     # -- Generate query --------------------------------------------
#     queries = generate_queries(df, n_total=n)

#     # -- Simpan & tampilkan ----------------------------------------
#     save_json(queries, OUTPUT_JSON)
#     print_statistics(queries)
#     print_sample(queries, n=5)

#     log("=" * 55)
#     log(f"DONE! {len(queries)} query uji tersimpan -> {OUTPUT_JSON}")
#     log("=" * 55)


# if __name__ == "__main__":
#     main()

# """
# queryUji.py
# ===========
# Pembentukan Query Uji untuk Eksperimen Search Suggestion

# Justifikasi:
# - Jumlah query  : Rumus Slovin (Nurjannah Syakrani & Naufal Athaya, JACOST 2025)
# - Kategori query: Taksonomi QAC Krishnan et al. (ADCS 2017)
#     * PREFIX_TITLE  → Mode 1 (Prefix Match),          Model LR
#     * PREFIX_AUTHOR → Mode 1 (Prefix Match),          Model LR
#     * PARTIAL       → Mode 3 (Pattern Match),         Model LR
#     * TYPO          → Mode 4 (Relaxed Pattern Match), Model GEN
# """

# import pandas as pd
# import random
# import json
# import math
# import os
# from datetime import datetime

# # ─────────────────────────────────────────────────────────────────
# # KONFIGURASI PATH  →  sesuaikan dengan struktur folder kamu
# # ─────────────────────────────────────────────────────────────────
# CSV_FILE    = "../hasil_books2.csv"          # ganti dengan path CSV kamu
# OUTPUT_JSON = "../test_queries_final.json"   # output query uji
# LOG_FILE    = "../query_generation.log"

# # ─────────────────────────────────────────────────────────────────
# # PROPORSI KATEGORI
# # ─────────────────────────────────────────────────────────────────
# PROPORSI = {
#     "PREFIX_TITLE" : 0.30,   # 30%
#     "PREFIX_AUTHOR": 0.20,   # 20%
#     "PARTIAL"      : 0.30,   # 30%
#     "TYPO"         : 0.20,   # 20%
# }

# # ─────────────────────────────────────────────────────────────────
# # QWERTY NEIGHBORS  (untuk simulasi typo Model GEN)
# # Sumber: Krishnan et al. (2017) Section 5
# # ─────────────────────────────────────────────────────────────────
# QWERTY = {
#     'a':'sqwz',  'b':'vghn',  'c':'xdfv',  'd':'srfce',
#     'e':'wrsdf', 'f':'dcvgt', 'g':'ftbhy', 'h':'gynj',
#     'i':'ujko',  'j':'hunkm', 'k':'jilm',  'l':'kop',
#     'm':'njk',   'n':'bhjm',  'o':'iklp',  'p':'ol',
#     'q':'wa',    'r':'edft',  's':'awedxz','t':'rfgy',
#     'u':'yhji',  'v':'cfgb',  'w':'qase',  'x':'zsdc',
#     'y':'tghu',  'z':'asx',
# }

# # =================================================================
# # 1. RUMUS SLOVIN
# #    Referensi: Nurjannah Syakrani & Naufal Athaya (2025)
# #               JACOST Vol.6 No.1 — Persamaan (4)
# #    n = N / (1 + N * e²)
# # =================================================================
# def slovin(N: int, e: float = 0.10) -> int:
#     """
#     Menghitung ukuran sampel menggunakan rumus Slovin.

#     Parameter
#     ----------
#     N : int   — jumlah populasi (total records dataset)
#     e : float — margin error (default 0.10 = 10%)

#     Return
#     ------
#     n : int   — jumlah query uji yang harus dibentuk
#     """
#     n = N / (1 + N * (e ** 2))
#     return math.ceil(n)


# # =================================================================
# # 2. MODEL LR — PREFIX  (Mode 1, Krishnan et al. 2017)
# #    Mengambil potongan dari bagian AWAL teks (kiri ke kanan).
# #    Merepresentasikan pengguna yang mengetik kata secara bertahap.
# # =================================================================
# def generate_prefix_LR(text: str) -> str:
#     """
#     Model LR dari Krishnan et al. (2017) Section 5, Tabel 4.
#     Append karakter dari kiri ke kanan → ambil prefix 30-50%
#     dari total kata.

#     Justifikasi panjang prefix:
#     - Paper menyebut minimum length = 3 karakter (Bast & Weber
#       dalam Krishnan et al. 2017 Section 5).
#     - 30-50% dipilih agar prefix tidak terlalu pendek (ambigu)
#       dan tidak terlalu panjang (terlalu mudah ditebak),
#       merepresentasikan pengguna yang baru mulai mengetik.

#     Contoh:
#       "the hunger games"    (3 kata) -> ambil 1 kata  -> "the"
#       "pride and prejudice" (3 kata) -> ambil 1-2 kata -> "pride and"
#       "gone with the wind"  (4 kata) -> ambil 1-2 kata -> "gone with"
#     """
#     words = text.strip().split()

#     if len(words) == 1:
#         # Untuk judul 1 kata: ambil 3-5 karakter pertama
#         cut = random.randint(3, min(5, len(words[0])))
#         return words[0][:cut]

#     # Hitung rentang 30-50% dari total kata, minimum 1
#     low  = max(1, math.ceil(len(words) * 0.30))
#     high = max(1, math.floor(len(words) * 0.50))

#     # Pastikan high tidak melebihi panjang - 1 (tidak ambil semua)
#     high = min(high, len(words) - 1)

#     # Jika low > high karena pembulatan, samakan
#     if low > high:
#         low = high

#     n_take = random.randint(low, high)
#     return " ".join(words[:n_take])


# # =================================================================
# # 3. MODEL LR — PARTIAL  (Mode 3, Krishnan et al. 2017)
# #    Mengambil kata dari posisi TENGAH atau AKHIR judul.
# #    Merepresentasikan pengguna yang hanya ingat sebagian kata.
# # =================================================================
# def generate_partial_LR(text: str) -> str:
#     """
#     Model LR untuk Mode 3 (Pattern Match).
#     Kata diambil bukan dari posisi awal (index > 0),
#     sehingga tidak bisa ditangkap prefix match biasa.

#     Aturan:
#     - ≤2 kata : kembalikan kata terakhir
#     - ≥3 kata : pilih 1–2 kata mulai dari index ≥ 1
#     """
#     words = text.strip().split()

#     if len(words) <= 2:
#         return words[-1]

#     # Mulai dari index 1 (bukan awal)
#     start = random.randint(1, len(words) - 1)
#     end   = min(start + random.randint(0, 1), len(words) - 1)
#     return " ".join(words[start:end + 1])


# # =================================================================
# # 4. MODEL GEN — TYPO  (Mode 4, Krishnan et al. 2017)
# #    Simulasi kesalahan ketik berdasarkan kedekatan karakter
# #    pada layout QWERTY (distribusi Gaussian, σ = 0.19).
# #    P_append = 0.80, P_delete = 0.04 (Section 5)
# # =================================================================
# def generate_typo_GEN(text: str, typo_prob: float = 0.25) -> str:
#     """
#     Model GEN dari Krishnan et al. (2017) Section 5.
#     Mensimulasikan typo ringan pada 1–2 kata dalam query.

#     typo_prob = probabilitas tiap karakter mengalami typo.
#     Nilai 0.25 menghasilkan rata-rata 1–2 typo per query
#     (masih menyerupai kata asli, sesuai batasan TA).
#     """
#     words = text.strip().split()

#     n_typo_words  = random.randint(1, min(2, len(words)))
#     idx_to_typo   = set(random.sample(range(len(words)), n_typo_words))

#     result = []
#     for i, word in enumerate(words):
#         if i in idx_to_typo:
#             new_chars = []
#             for ch in word:
#                 if ch.isalpha() and ch in QWERTY and random.random() < typo_prob:
#                     new_chars.append(random.choice(QWERTY[ch]))
#                 else:
#                     new_chars.append(ch)
#             result.append("".join(new_chars))
#         else:
#             result.append(word)

#     return " ".join(result)


# def generate_queries(df: pd.DataFrame, n_total: int,
#                      seed: int = 42) -> list[dict]:
#     """
#     Membentuk query uji sesuai jumlah Slovin dan 4 kategori.

#     Parameter
#     ----------
#     df      : DataFrame dengan kolom 'title' dan 'author'
#     n_total : jumlah total query (hasil Slovin)
#     seed    : random seed untuk reprodusibilitas
#     """
#     random.seed(seed)
#     books = df.to_dict('records')

#     n_pt = math.ceil(n_total * PROPORSI["PREFIX_TITLE"])
#     n_pa = math.ceil(n_total * PROPORSI["PREFIX_AUTHOR"])
#     n_par= math.ceil(n_total * PROPORSI["PARTIAL"])
#     n_ty = n_total - n_pt - n_pa - n_par   

#     print(f"\n{'='*55}")
#     print(f"  DISTRIBUSI QUERY (Total Slovin = {n_total})")
#     print(f"{'='*55}")
#     print(f"  PREFIX_TITLE   (Mode 1, LR ) : {n_pt:3d} query (30%)")
#     print(f"  PREFIX_AUTHOR  (Mode 1, LR ) : {n_pa:3d} query (20%)")
#     print(f"  PARTIAL        (Mode 3, LR ) : {n_par:3d} query (30%)")
#     print(f"  TYPO           (Mode 4, GEN) : {n_ty:3d} query (20%)")
#     print(f"{'='*55}\n")

#     queries    = []
#     used_texts = set() 

#     def try_add(query_text, qtype, book, difficulty, mode_ref) -> bool:
#         """Tambahkan query jika belum duplikat dan tidak kosong."""
#         qt = query_text.strip().lower()
#         if not qt or qt in used_texts:
#             return False
#         used_texts.add(qt)
#         queries.append({
#             "query_id"   : f"Q{len(queries)+1:03d}",
#             "query_text" : query_text.strip(),
#             "query_type" : qtype,
#             "title"      : book["title"],
#             "author"     : book.get("author", "unknown"),
#         })
#         return True

#     pool  = [b for b in books
#              if isinstance(b["title"], str)
#              and len(b["title"].split()) >= 2]
#     random.shuffle(pool)
#     count = 0
#     for book in pool:
#         if count >= n_pt:
#             break
#         q = generate_prefix_LR(book["title"])
#         if try_add(q, "PREFIX_TITLE", book, "EASY",
#                    "Mode 1 – Prefix Match (Krishnan et al., 2017), Model LR"):
#             count += 1
#     print(f"  ✓ PREFIX_TITLE   : {count} query terbentuk")

#     pool  = [b for b in books
#              if isinstance(b.get("author"), str)
#              and b["author"].strip()]
#     random.shuffle(pool)
#     count = 0
#     for book in pool:
#         if count >= n_pa:
#             break
#         q = generate_prefix_LR(book["author"])
#         if try_add(q, "PREFIX_AUTHOR", book, "EASY",
#                    "Mode 1 – Prefix Match (Krishnan et al., 2017), Model LR"):
#             count += 1
#     print(f"  ✓ PREFIX_AUTHOR  : {count} query terbentuk")

#     pool  = [b for b in books
#              if isinstance(b["title"], str)
#              and len(b["title"].split()) >= 3]
#     random.shuffle(pool)
#     count = 0
#     for book in pool:
#         if count >= n_par:
#             break
#         q = generate_partial_LR(book["title"])
#         if try_add(q, "PARTIAL", book, "MEDIUM",
#                    "Mode 3 – Pattern Match (Krishnan et al., 2017), Model LR"):
#             count += 1
#     print(f"  ✓ PARTIAL        : {count} query terbentuk")

#     pool  = [b for b in books
#              if isinstance(b["title"], str)
#              and len(b["title"].split()) >= 2]
#     random.shuffle(pool)
#     count = 0
#     for book in pool:
#         if count >= n_ty:
#             break
#         q = generate_typo_GEN(book["title"])
        
#         if q.lower() != book["title"].lower():
#             if try_add(q, "TYPO", book, "HARD",
#                        "Mode 4 – Relaxed Pattern Match (Krishnan et al., 2017), Model GEN"):
#                 count += 1
#     print(f"  ✓ TYPO           : {count} query terbentuk")

#     return queries

# def log(msg: str):
#     ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     line = f"[{ts}] {msg}"
#     print(line)
#     with open(LOG_FILE, "a", encoding="utf-8") as f:
#         f.write(line + "\n")

# def save_json(data: list, path: str):
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=2, ensure_ascii=False)
#     log(f"✓ Tersimpan → {path}  ({len(data)} queries)")

# def print_sample(queries: list, n: int = 5):
#     print(f"\n{'='*55}")
#     print(f"  PREVIEW {n} QUERY PERTAMA")
#     print(f"{'='*55}")
#     for q in queries[:n]:
#         print(f"\n  [{q['query_id']}] \"{q['query_text']}\"")
#         print(f"       Type  : {q['query_type']}")
#         print(f"       Title : {q['title']}")
#         print(f"       Author: {q['author']}")

# def print_statistics(queries: list):
#     from collections import Counter
#     types = Counter(q["query_type"] for q in queries)

#     print(f"\n{'='*55}")
#     print(f"  STATISTIK QUERY UJI")
#     print(f"{'='*55}")
#     print(f"  Total : {len(queries)} queries\n")
#     print("  TIPE:")
#     for t in ["PREFIX_TITLE","PREFIX_AUTHOR","PARTIAL","TYPO"]:
#         c   = types[t]
#         pct = c / len(queries) * 100
#         bar = "█" * int(pct / 5)
#         print(f"    {t:16} {c:3}  ({pct:5.1f}%)  {bar}")


# def main():
#     with open(LOG_FILE, "w") as f:
#         f.write("")

#     log("=" * 55)
#     log("START: Pembentukan Query Uji")
#     log("=" * 55)

#     log(f"Loading CSV: {CSV_FILE}")
#     df = pd.read_csv(CSV_FILE, usecols=["title", "author"])
#     df = df.dropna(subset=["title"])
#     df["title"]  = df["title"].str.lower().str.strip()
#     df["author"] = df["author"].str.lower().str.strip()
#     df = df.drop_duplicates(subset=["title"])
#     df = df.reset_index(drop=True)
#     N  = len(df)
#     log(f"✓ Dataset: {N:,} records unik")

#     e = 0.10  
#     n = slovin(N, e)
#     log(f"✓ Rumus Slovin → N={N:,}, e={e*100:.0f}% → n={n}")
#     print(f"\n  Perhitungan Slovin:")
#     print(f"  n = {N} / (1 + {N} × {e}²)")
#     print(f"  n = {N} / (1 + {N * e**2:.2f})")
#     print(f"  n = {N} / {1 + N * e**2:.2f}")
#     print(f"  n = {N / (1 + N * e**2):.4f} → dibulatkan ke atas = {n}")

#     queries = generate_queries(df, n_total=n)

#     save_json(queries, OUTPUT_JSON)
#     print_statistics(queries)
#     print_sample(queries, n=5)

#     log("=" * 55)
#     log(f"DONE! {len(queries)} query uji tersimpan → {OUTPUT_JSON}")
#     log("=" * 55)


# if __name__ == "__main__":
#     main()

import pandas as pd
import random
import json
import math
import os
from datetime import datetime

CSV_FILE    = "../hasil_books2.csv"          # ganti dengan path CSV kamu
OUTPUT_JSON = "../v3test_queries_final.json"   # output query uji
LOG_FILE    = "../query_generation.log"

PROPORSI = {
    "PREFIX_TITLE" : 0.30,   # 30%
    "PREFIX_AUTHOR": 0.20,   # 20%
    "PARTIAL"      : 0.30,   # 30%
    "TYPO"         : 0.20,   # 20%
}

QWERTY = {
    'a':'sqwz',  'b':'vghn',  'c':'xdfv',  'd':'srfce',
    'e':'wrsdf', 'f':'dcvgt', 'g':'ftbhy', 'h':'gynj',
    'i':'ujko',  'j':'hunkm', 'k':'jilm',  'l':'kop',
    'm':'njk',   'n':'bhjm',  'o':'iklp',  'p':'ol',
    'q':'wa',    'r':'edft',  's':'awedxz','t':'rfgy',
    'u':'yhji',  'v':'cfgb',  'w':'qase',  'x':'zsdc',
    'y':'tghu',  'z':'asx',
}

def slovin(N: int, e: float = 0.10) -> int:
    n = N / (1 + N * (e ** 2))
    return math.ceil(n)


def generate_prefix_LR(text: str, for_author: bool = False) -> str:
    """
    Justifikasi panjang prefix:
    - Paper menyebut minimum length = 3 karakter (Bast & Weber
      dalam Krishnan et al. 2017 Section 5).
    - Minimum 2 kata diterapkan agar query tidak terlalu ambigu.
    - for_author=True: skip author dengan token inisial tunggal
      (< 3 karakter) agar query cukup spesifik ke pengarang.

    FIX PREFIX_TITLE:
      - Pool wajib judul >= 3 kata (difilter di generate_queries).
      - Ambil 2 kata (minimum), sehingga selalu ada sisa >= 1 kata
        yang belum ditulis → valid sebagai query prefix Mode 1.

    FIX PREFIX_AUTHOR:
      - Skip jika salah satu token nama adalah inisial tunggal
        (< 3 karakter), karena "j m" terlalu ambigu dan tidak bisa
        digunakan untuk menemukan satu pengarang secara spesifik.
      - Selalu ambil nama lengkap (semua token) agar spesifik.

    Contoh PREFIX_TITLE:
      "the hunger games"    (3 kata) -> "the hunger"
      "pride and prejudice" (3 kata) -> "pride and"
      "gone with the wind"  (4 kata) -> "gone with"

    Contoh PREFIX_AUTHOR:
      "andrea zuvich"   (2 kata) -> "andrea zuvich"
      "derrick jensen"  (2 kata) -> "derrick jensen"
    """
    words = text.strip().split()

    if len(words) == 1:
        # 1 kata: ambil 3-5 karakter pertama
        cut = random.randint(3, min(5, len(words[0])))
        return words[0][:cut]

    if for_author:
        # FIX: tolak jika ada token inisial (< 3 karakter)
        # mis. "j m barrie" ditolak karena "j" dan "m" terlalu ambigu
        if any(len(w) < 3 for w in words):
            return ""   # sinyal ke pemanggil untuk skip buku ini

        # Ambil semua token nama (nama lengkap) agar spesifik
        # Jika nama panjang (>= 3 token), ambil 2 token pertama saja
        n_take = min(2, len(words))
        return " ".join(words[:n_take])

    # Title: minimum 2 kata, rentang 30-50% dari total kata
    # Pool dijamin >= 3 kata dari generate_queries, jadi aman ambil 2
    low  = max(2, math.ceil(len(words) * 0.30))
    high = max(2, math.floor(len(words) * 0.50))

    # Pastikan high tidak melebihi panjang - 1 (tidak ambil semua)
    high = min(high, len(words) - 1)

    # Jika low > high karena pembulatan, samakan
    if low > high:
        low = high

    n_take = random.randint(low, high)
    return " ".join(words[:n_take])


def generate_partial_LR(text: str) -> str:
    """
    FIX PARTIAL:
    - Pool dijamin >= 4 kata dari generate_queries.
    - Selalu ambil MINIMAL 2 kata (start..start+1) agar query
      tidak jadi 1 kata stop word yang terlalu ambigu
      (mis. "of", "in", "and", "the" -> ribuan hasil).
    - Hindari mengambil stop word tunggal sebagai hasil akhir.

    Aturan:
    - Mulai dari index 1 (bukan awal), ambil 2 kata (tidak 1).
    - Jika index terlalu dekat ujung, geser ke kiri.
    """
    STOP_WORDS = {
        "a","an","the","and","or","of","in","on","at","to",
        "for","by","with","as","is","it","its","be","was",
        "are","from","this","that","not","but","if","up"
    }

    words = text.strip().split()

    # Pool dijamin >= 4 kata dari generate_queries,
    # tapi ada safeguard di sini juga
    if len(words) < 3:
        # Kembalikan kata terakhir yang bukan stop word
        for w in reversed(words):
            if w.lower() not in STOP_WORDS:
                return w
        return words[-1]

    # Mulai dari index 1, ambil 2 kata
    # Pastikan start + 1 masih dalam batas
    max_start = len(words) - 2   # minimal 2 kata setelah start
    start = random.randint(1, max(1, max_start))
    end   = min(start + 1, len(words) - 1)   # selalu ambil 2 kata

    chunk = words[start:end + 1]

    # Jika hasil hanya 1 kata dan itu stop word, coba geser
    if len(chunk) == 1 and chunk[0].lower() in STOP_WORDS:
        # Cari 2 kata yang bukan stop word dari posisi lain
        for s in range(1, len(words) - 1):
            candidate = words[s:s + 2]
            if not all(w.lower() in STOP_WORDS for w in candidate):
                return " ".join(candidate)
        # Fallback: ambil 2 kata terakhir
        return " ".join(words[-2:])

    return " ".join(chunk)


def generate_typo_GEN(text: str, typo_prob: float = 0.25,
                      max_typo_chars: int = 2) -> str:
    """

    typo_prob     = probabilitas tiap karakter mengalami typo.
    max_typo_chars = batas atas jumlah karakter yang boleh ditypo
                     per query (default 2), agar query masih
                     menyerupai kata asli dan bisa ditemukan sistem.

    FIX TYPO:
    - Stop word pendek ("the", "a", "an", "of", "in", dll.)
      TIDAK boleh ditypo karena perubahan sekecil apapun
      akan mengubahnya menjadi string yang tidak relevan sama sekali
      (mis. "the" -> "tyw" membuat fuzzy search gagal total).
    - Typo hanya dilakukan pada kata konten (kata substantif).

    Contoh yang diharapkan:
      "the hunger games" -> "the hunfer games"  (typo di "hunger", bukan "the")
      "above world"      -> "abpve world"        (1 typo)
    """
    STOP_WORDS = {
        "a","an","the","and","or","of","in","on","at","to",
        "for","by","with","as","is","it","its","be","was",
        "are","from","this","that","not","but","if","up","i"
    }

    words = text.strip().split()

    # Kandidat kata yang boleh ditypo: bukan stop word
    eligible = [i for i, w in enumerate(words)
                if w.lower() not in STOP_WORDS and len(w) >= 3]

    # Jika tidak ada kata eligible, fallback: typo kata terpanjang
    if not eligible:
        eligible = [max(range(len(words)), key=lambda i: len(words[i]))]

    # Pilih 1-2 kata dari yang eligible
    n_typo_words = random.randint(1, min(2, len(eligible)))
    idx_to_typo  = set(random.sample(eligible, n_typo_words))

    result           = []
    total_typo_count = 0   # hitung total karakter yang sudah ditypo

    for i, word in enumerate(words):
        if i in idx_to_typo and total_typo_count < max_typo_chars:
            new_chars = []
            for ch in word:
                # Hentikan typo jika sudah capai batas max
                if (total_typo_count < max_typo_chars
                        and ch.isalpha()
                        and ch in QWERTY
                        and random.random() < typo_prob):
                    new_chars.append(random.choice(QWERTY[ch]))
                    total_typo_count += 1
                else:
                    new_chars.append(ch)
            result.append("".join(new_chars))
        else:
            result.append(word)

    return " ".join(result)


def generate_queries(df: pd.DataFrame, n_total: int,
                     seed: int = 42) -> list[dict]:
    """
    df      : DataFrame dengan kolom 'title' dan 'author'
    n_total : jumlah total query (hasil Slovin)
    seed    : random seed untuk reprodusibilitas
    """
    random.seed(seed)
    books = df.to_dict('records')

    # Hitung jumlah per kategori
    n_pt  = math.ceil(n_total * PROPORSI["PREFIX_TITLE"])
    n_pa  = math.ceil(n_total * PROPORSI["PREFIX_AUTHOR"])
    n_par = math.ceil(n_total * PROPORSI["PARTIAL"])
    n_ty  = n_total - n_pt - n_pa - n_par   # sisa ke TYPO

    print(f"\n{'='*55}")
    print(f"  DISTRIBUSI QUERY (Total Slovin = {n_total})")
    print(f"{'='*55}")
    print(f"  PREFIX_TITLE   (Mode 1, LR ) : {n_pt:3d} query (30%)")
    print(f"  PREFIX_AUTHOR  (Mode 1, LR ) : {n_pa:3d} query (20%)")
    print(f"  PARTIAL        (Mode 3, LR ) : {n_par:3d} query (30%)")
    print(f"  TYPO           (Mode 4, GEN) : {n_ty:3d} query (20%)")
    print(f"{'='*55}\n")

    queries    = []
    used_texts = set()   # cegah duplikat

    def try_add(query_text, qtype, book, difficulty, mode_ref) -> bool:
        qt = query_text.strip().lower()
        if not qt or qt in used_texts:
            return False
        used_texts.add(qt)
        queries.append({
            "query_id"   : f"Q{len(queries)+1:03d}",
            "query_text" : query_text.strip(),
            "query_type" : qtype,
            "title"      : book["title"],
            "author"     : book.get("author", "unknown"),
        })
        return True

    # -- PREFIX TITLE (Mode 1, LR) ---------------------------------
    # FIX: filter judul >= 3 kata agar selalu bisa ambil 2 kata prefix
    #      dengan sisa >= 1 kata (valid Mode 1).
    #      Judul 2 kata menghasilkan prefix 1 kata yang terlalu ambigu
    #      (mis. "blood" -> ratusan buku dengan awalan "blood").
    pool  = [b for b in books
             if isinstance(b["title"], str)
             and len(b["title"].split()) >= 3]
    random.shuffle(pool)
    count = 0
    for book in pool:
        if count >= n_pt:
            break
        q = generate_prefix_LR(book["title"], for_author=False)
        if try_add(q, "PREFIX_TITLE", book, "EASY",
                   "Mode 1 - Prefix Match (Krishnan et al., 2017), Model LR"):
            count += 1
    print(f"  OK PREFIX_TITLE   : {count} query terbentuk")

    # -- PREFIX AUTHOR (Mode 1, LR) --------------------------------
    # FIX: filter author >= 2 kata, AND skip jika generate_prefix_LR
    #      mengembalikan "" (tandanya ada token inisial tunggal).
    #      Inisial tunggal seperti "j m" tidak spesifik ke 1 pengarang.
    pool  = [b for b in books
             if isinstance(b.get("author"), str)
             and len(b["author"].strip().split()) >= 2]
    random.shuffle(pool)
    count = 0
    for book in pool:
        if count >= n_pa:
            break
        q = generate_prefix_LR(book["author"], for_author=True)
        if not q:   # skip: ada token inisial tunggal
            continue
        if try_add(q, "PREFIX_AUTHOR", book, "EASY",
                   "Mode 1 - Prefix Match (Krishnan et al., 2017), Model LR"):
            count += 1
    print(f"  OK PREFIX_AUTHOR  : {count} query terbentuk")

    # -- PARTIAL (Mode 3, LR) --------------------------------------
    # FIX: filter judul >= 4 kata agar generate_partial_LR selalu
    #      bisa mengambil 2 kata dari posisi tengah/akhir tanpa
    #      terpaksa mengambil stop word tunggal sebagai query.
    pool  = [b for b in books
             if isinstance(b["title"], str)
             and len(b["title"].split()) >= 4]
    random.shuffle(pool)
    count = 0
    for book in pool:
        if count >= n_par:
            break
        q = generate_partial_LR(book["title"])
        if try_add(q, "PARTIAL", book, "MEDIUM",
                   "Mode 3 - Pattern Match (Krishnan et al., 2017), Model LR"):
            count += 1
    print(f"  OK PARTIAL        : {count} query terbentuk")

    # -- TYPO (Mode 4, GEN) ----------------------------------------
    # FIX: max_typo_chars=2 agar tidak lebih dari 2 karakter berubah
    pool  = [b for b in books
             if isinstance(b["title"], str)
             and len(b["title"].split()) >= 2]
    random.shuffle(pool)
    count = 0
    for book in pool:
        if count >= n_ty:
            break
        q = generate_typo_GEN(book["title"], typo_prob=0.25, max_typo_chars=2)
        # Pastikan hasil typo berbeda dari aslinya
        if q.lower() != book["title"].lower():
            if try_add(q, "TYPO", book, "HARD",
                       "Mode 4 - Relaxed Pattern Match (Krishnan et al., 2017), Model GEN"):
                count += 1
    print(f"  OK TYPO           : {count} query terbentuk")

    return queries


# =================================================================
# 6. LOGGING & SIMPAN
# =================================================================
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
    print(f"\n{'='*55}")
    print(f"  PREVIEW {n} QUERY PERTAMA")
    print(f"{'='*55}")
    for q in queries[:n]:
        print(f"\n  [{q['query_id']}] \"{q['query_text']}\"")
        print(f"       Type  : {q['query_type']}")
        print(f"       Title : {q['title']}")
        print(f"       Author: {q['author']}")

def print_statistics(queries: list):
    from collections import Counter
    types = Counter(q["query_type"] for q in queries)

    print(f"\n{'='*55}")
    print(f"  STATISTIK QUERY UJI")
    print(f"{'='*55}")
    print(f"  Total : {len(queries)} queries\n")
    print("  TIPE:")
    for t in ["PREFIX_TITLE","PREFIX_AUTHOR","PARTIAL","TYPO"]:
        c   = types[t]
        pct = c / len(queries) * 100
        bar = "X" * int(pct / 5)
        print(f"    {t:16} {c:3}  ({pct:5.1f}%)  {bar}")


# =================================================================
# MAIN
# =================================================================
def main():
    # Reset log
    with open(LOG_FILE, "w") as f:
        f.write("")

    log("=" * 55)
    log("START: Pembentukan Query Uji")
    log("=" * 55)

    # -- Load CSV --------------------------------------------------
    log(f"Loading CSV: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE, usecols=["title", "author"])
    df = df.dropna(subset=["title"])
    df["title"]  = df["title"].str.lower().str.strip()
    df["author"] = df["author"].fillna("").str.lower().str.strip()
    df = df.drop_duplicates(subset=["title"])
    df = df.reset_index(drop=True)
    N  = len(df)
    log(f"Dataset: {N:,} records unik")

    # -- Hitung n (Rumus Slovin) -----------------------------------
    e = 0.10   # margin error 10%
    n = slovin(N, e)
    log(f"Rumus Slovin -> N={N:,}, e={e*100:.0f}% -> n={n}")
    print(f"\n  Perhitungan Slovin:")
    print(f"  n = {N} / (1 + {N} x {e}^2)")
    print(f"  n = {N} / (1 + {N * e**2:.2f})")
    print(f"  n = {N} / {1 + N * e**2:.2f}")
    print(f"  n = {N / (1 + N * e**2):.4f} -> dibulatkan ke atas = {n}")

    # -- Generate query --------------------------------------------
    queries = generate_queries(df, n_total=n)

    # -- Simpan & tampilkan ----------------------------------------
    save_json(queries, OUTPUT_JSON)
    print_statistics(queries)
    print_sample(queries, n=5)

    log("=" * 55)
    log(f"DONE! {len(queries)} query uji tersimpan -> {OUTPUT_JSON}")
    log("=" * 55)


if __name__ == "__main__":
    main()