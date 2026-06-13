"""
queryUji_qac.py
===============
Pembentukan Query Uji untuk Eksperimen Search Suggestion

Justifikasi:
- Jumlah query  : Rumus Slovin (Nurjannah Syakrani & Naufal Athaya, JACOST 2025)
- Kategori query: Taksonomi QAC Krishnan et al. (ADCS 2017)

    Label QAC               Mode  Teknik                   Model
    ─────────────────────── ───── ──────────────────────── ────────
    MODE1_PREFIX_TITLE      1     Prefix Match (judul)     LR
    MODE1_PREFIX_AUTHOR     1     Prefix Match (pengarang) LR
    MODE2_MULTI_TERM_PREFIX 2     Multi-term Prefix Match  LR
    MODE3_PARTIAL           3     Pattern Match (substr.)  LR
    MODE4_TYPO              4     Relaxed Pattern Match    GEN (typo)

Referensi utama:
  Krishnan, U., Moffat, A., & Zobel, J. (2017).
  A Taxonomy of Query Auto Completion Modes.
  ADCS 2017, Brisbane, QLD, Australia.
  https://doi.org/10.1145/3166072.3166081

Mode yang TIDAK diimplementasikan:
  - Mode 0 (Exact Match): trivial, tidak relevan untuk skenario
    autocomplete yang berfokus pada query parsial.
  - GEN penuh (Append/Delete/Jump per keystroke): tidak diperlukan
    karena penelitian ini tidak mensimulasikan interaksi per keystroke.
    Konsep GEN diadopsi sebagian pada Mode 4 (simulasi typo keyboard).
"""

import pandas as pd
import random
import json
import math
import os
from datetime import datetime

# ─────────────────────────────────────────────────────────────────
# KONFIGURASI PATH  →  sesuaikan dengan struktur folder kamu
# ─────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
CSV_FILE    = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "DATASET", "hasil_books2.csv"))
OUTPUT_JSON = "./v9test_queries_final.json"
LOG_FILE    = "./query_generation_qac.log"

# ─────────────────────────────────────────────────────────────────
# PROPORSI KATEGORI (Mode 1–4)
#
# Krishnan et al. (2017) TIDAK menetapkan proporsi antar mode.
# Paper hanya mendefinisikan mode secara formal dan mengukur
# performa masing-masing mode secara independen (Section 5).
#
# Proporsi ditetapkan dengan distribusi SERAGAM antar mode,
# yaitu masing-masing 25% dari total query. Mode 1 dibagi dua
# sub-label (title & author) sehingga masing-masing 12.5%.
#
# Justifikasi distribusi seragam:
#   Tidak ada dasar empiris dari Krishnan et al. (2017) maupun
#   literatur lain yang menetapkan proporsi antar mode QAC.
#   Distribusi seragam dipilih agar setiap mode mendapat representasi
#   setara sehingga perbandingan antar mode tidak bias secara
#   statistik akibat perbedaan jumlah sampel (prinsip balanced
#   experimental design). Mode 3 dan 4 diberi porsi lebih kecil
#   karena pool kandidatnya lebih terbatas secara teknis.
# ─────────────────────────────────────────────────────────────────
PROPORSI = {
    "MODE1_PREFIX_TITLE"     : 0.20,    # Mode 1, LR (title)
    "MODE1_PREFIX_AUTHOR"    : 0.20,    # Mode 1, LR (author)
    "MODE2_MULTI_TERM_PREFIX": 0.20,    # Mode 2, LR
    "MODE3_PARTIAL"          : 0.20,   #Mode 3, LR
    "MODE4_TYPO"             : 0.20,   # Mode 4, GEN
}

# ─────────────────────────────────────────────────────────────────
# QWERTY ADJACENCY MAP  (untuk simulasi typo Mode 4 / Model GEN)
# Sumber: Krishnan et al. (2017) Section 5 — keyboard adjacency
# Digunakan sebagai dasar substitution error (prioritas utama,
# sesuai temuan empiris Kernighan et al., 1990).
# ─────────────────────────────────────────────────────────────────
QWERTY = {
    'a': 'sqwz',  'b': 'vghn',  'c': 'xdfv',  'd': 'srfce',
    'e': 'wrsdf', 'f': 'dcvgt', 'g': 'ftbhy', 'h': 'gynj',
    'i': 'ujko',  'j': 'hunkm', 'k': 'jilm',  'l': 'kop',
    'm': 'njk',   'n': 'bhjm',  'o': 'iklp',  'p': 'ol',
    'q': 'wa',    'r': 'edft',  's': 'awedxz','t': 'rfgy',
    'u': 'yhji',  'v': 'cfgb',  'w': 'qase',  'x': 'zsdc',
    'y': 'tghu',  'z': 'asx',
}

# Stop words yang tidak boleh ditypo / tidak berguna sebagai token
STOP_WORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to",
    "for", "by", "with", "as", "is", "it", "its", "be", "was",
    "are", "from", "this", "that", "not", "but", "if", "up", "i",
    "do", "so", "no", "my", "we", "he", "she", "they", "us",
}


# =================================================================
# 1. RUMUS SLOVIN
#    Referensi: Nurjannah Syakrani & Naufal Athaya (2025)
#               JACOST Vol.6 No.1 — Persamaan (4)
#    n = N / (1 + N * e²)
# =================================================================
def slovin(N: int, e: float = 0.10) -> int:
    """
    Menghitung ukuran sampel menggunakan rumus Slovin.

    Parameter
    ----------
    N : int   — jumlah populasi (total records dataset)
    e : float — margin error (default 0.10 = 10%)

    Return
    ------
    n : int   — jumlah query uji yang harus dibentuk
    """
    return math.ceil(N / (1 + N * (e ** 2)))


# =================================================================
# 2. MODE 1 — PREFIX MATCH  (Krishnan et al., 2017 — Eq. 2)
#    Model LR: karakter ditambahkan dari kiri ke kanan.
#    Prefix diambil dari awal string target (judul atau pengarang).
#
#    Definisi formal (paper, Section 3):
#      PrefixMatch(P) = { Ti | Ti ∈ T ∧ P ∈ Prefix(Ti) }
#
#    Sub-label:
#      MODE1_PREFIX_TITLE  → prefix dari judul buku
#      MODE1_PREFIX_AUTHOR → prefix dari nama pengarang
# =================================================================
def generate_mode1_prefix_title(title: str) -> str:
    """
    Model LR untuk Mode 1 — PREFIX TITLE.
    Mengambil awalan (prefix) dari judul buku secara bertahap
    dari kiri ke kanan (sesuai LR model, Section 5, Table 4).

    Strategi:
    - Judul 1 kata  : ambil 3–5 karakter pertama (minimum 3 char,
                      sesuai batas bawah Bast & Weber dalam paper).
    - Judul multi-kata: ambil 30–50% kata dari awal, minimal 2 kata,
                      tidak mengambil seluruh judul (harus ada sisa
                      ≥1 kata yang belum diketik, agar benar-benar
                      merepresentasikan query parsial Mode 1).

    Contoh:
      "the hunger games"    → "the hunger"
      "pride and prejudice" → "pride and"
      "gone with the wind"  → "gone with"
      "dune"                → "dun"
    """
    words = title.strip().split()

    if len(words) == 1:
        cut = random.randint(3, min(5, len(words[0])))
        return words[0][:cut]

    # ── Anti-anomali: cari index kata konten pertama (non-stop word) ──
    # Judul seperti "the complete works", "the art of war" → prefix
    # 2 kata pertama berisi stop word / kata terlalu umum.
    # Solusi:
    # (A) Kata konten valid: bukan stop word DAN panjang >= 5
    #     (sehingga "art"(3), "war"(3), "girl"(4) tidak lolos).
    # (B) Prefix harus sertakan kata konten valid itu + minimal 1
    #     kata setelahnya agar query tidak berhenti di kata yang
    #     terlalu umum saja.
    # (C) Jika tidak ada kata konten valid (judul terlalu pendek),
    #     kembalikan "" → pool filter di generate_queries yang
    #     memastikan judul >= 3 kata akan mengurangi kasus ini.

    # Cari kata konten pertama (non-stop, len >= 5)
    first_content_idx = None
    for i, w in enumerate(words):
        if w.lower() not in STOP_WORDS and len(w) >= 5:
            first_content_idx = i
            break

    # Tidak ada kata konten valid → tolak (terlalu umum/pendek)
    if first_content_idx is None:
        return ""

    # min_take: sertakan kata konten itu + setidaknya 1 kata setelahnya
    # agar prefix tidak berhenti di kata tunggal yang umum.
    # Tapi kalau kata konten ada di posisi terakhir-1, cukup sertakan dia.
    # min_take: sertakan kata konten (first_content_idx) + 1 kata sesudahnya
    # agar prefix tidak berhenti hanya di kata konten itu sendiri.
    # Pengecualian: jika kata konten ada di posisi terakhir (edge case
    # seperti "to kill a mockingbird" → konten di idx 3 dari 4 kata),
    # boleh include dia sebagai kata terakhir prefix.
    min_take = first_content_idx + 1   # sertakan kata konten itu sendiri

    # Jika min_take melebihi panjang judul, tarik mundur ke len-1
    # tapi pastikan kata konten tetap masuk (idx = len-1 → min_take = len)
    # → dalam kasus ini boleh ambil sampai akhir (kata konten = kata terakhir)
    if min_take > len(words):
        min_take = len(words)   # include seluruh judul (rare case)

    max_take = min(
        max(min_take, math.floor(len(words) * 0.55)),
        len(words)              # boleh include kata terakhir jika perlu
    )
    # Tapi tidak boleh ambil SEMUA kata (harus query parsial):
    # kecuali jika judul pendek dan kata konten memang di akhir.
    if max_take >= len(words) and first_content_idx < len(words) - 1:
        max_take = len(words) - 1
    if min_take > max_take:
        max_take = min_take

    n_take = random.randint(min_take, max_take)
    prefix = " ".join(words[:n_take])

    # Validasi akhir: prefix harus mengandung >= 1 kata konten valid
    prefix_words = prefix.split()
    has_content = any(w not in STOP_WORDS and len(w) >= 5 for w in prefix_words)
    if not has_content:
        return ""

    return prefix


def generate_mode1_prefix_author(author: str) -> str:
    """
    Model LR untuk Mode 1 — PREFIX AUTHOR.
    Mengambil awalan nama pengarang dari kiri ke kanan.

    Strategi:
    - Tolak author dengan token inisial tunggal (< 3 karakter)
      karena "j m" atau "j k" terlalu ambigu untuk menemukan
      satu pengarang secara spesifik.
    - Ambil 2 token pertama nama (nama depan + nama belakang)
      agar prefix cukup spesifik.
    - Tambahkan variasi panjang prefix karakter pada token terakhir
      untuk mensimulasikan pengguna yang belum selesai mengetik
      (misalnya: "andrea z" bukan hanya "andrea zuvich").

    Contoh:
      "andrea zuvich"  → "andrea zu"   (prefix 2 token, token ke-2 dipotong)
      "derrick jensen" → "derrick jen"
    Return "" jika author tidak memenuhi syarat (pemanggil harus skip).
    """
    words = author.strip().split()

    # Tolak inisial tunggal
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


# =================================================================
# 3. MODE 2 — MULTI-TERM PREFIX MATCH  (Krishnan et al., 2017 — Eq. 3)
#    Model LR: beberapa token diambil dari kiri ke kanan secara
#    terpisah, lalu digabungkan menjadi query multi-kata.
#    Urutan token TIDAK harus sama dengan urutan di judul.
#
#    Definisi formal (paper, Section 3):
#      MultiTermPrefixMatch = { Ti | Ti ∈ T ∧ P* ⊆ ∪k Prefix(Ti^k) }
#
#    Contoh dari paper (Section 3):
#      "ste jo" → cocok dengan "steve jobs" DAN "joanne stewart"
#
#    Implementasi:
#    - Pilih 2–3 token berbeda dari judul (bukan stop word).
#    - Ambil prefix 30–70% dari masing-masing token.
#    - Gabungkan prefix-prefix tersebut dengan spasi.
#    - Token boleh diambil dari posisi mana saja (order bebas).
#
#    Contoh:
#      "deep learning with python"  → "dee pyt"
#      "the art of war"             → "ar wa"
#      "introduction to algorithms" → "int alg"
# =================================================================
def generate_mode2_multi_term_prefix(title: str) -> str:
    """
    Model LR untuk Mode 2 — MULTI-TERM PREFIX MATCH.
    Menghasilkan query dari potongan awalan beberapa token berbeda
    yang digabung, mencerminkan pengguna yang mengetik beberapa
    kata secara parsial dari berbagai bagian judul.
    """
    words = title.strip().split()

    # Filter token: bukan stop word, panjang >= 4
    # (minimal 4 agar prefix 50-70% masih >= 2 char dan tidak trivial)
    candidates = [w for w in words if w.lower() not in STOP_WORDS and len(w) >= 4]

    # Fallback: jika kandidat < 2, turunkan threshold ke >= 3
    if len(candidates) < 2:
        candidates = [w for w in words if w.lower() not in STOP_WORDS and len(w) >= 3]

    # Masih < 2 → tidak bisa bentuk multi-term prefix
    if len(candidates) < 2:
        return ""

    # Pilih 2 token berbeda secara acak (tanpa urutan)
    n_pick = min(2, len(candidates))
    picked = random.sample(candidates, n_pick)

    # Ambil prefix 40–70% dari setiap token yang dipilih
    # Minimum absolut: 3 karakter (mencegah prefix 1-2 char yang ambigu)
    prefixes = []
    for token in picked:
        min_cut = max(3, math.ceil(len(token) * 0.40))
        max_cut = max(3, math.ceil(len(token) * 0.70))
        if min_cut > len(token): min_cut = len(token)
        if max_cut > len(token): max_cut = len(token)
        if min_cut > max_cut:    min_cut = max_cut
        cut = random.randint(min_cut, max_cut)
        prefixes.append(token[:cut])

    # Validasi akhir: semua prefix harus >= 3 char
    if any(len(p) < 3 for p in prefixes):
        return ""

    return " ".join(prefixes)


# =================================================================
# 4. MODE 3 — PATTERN MATCH  (Krishnan et al., 2017 — Eq. 5)
#    Model LR: query berasal dari substring internal judul,
#    bukan hanya dari awal (tidak bisa ditangkap prefix match).
#
#    Definisi formal (paper, Section 3):
#      PatternMatch = { Ti | Ti ∈ T ∧ (∧ P^k ∈ P* Match(Ti, P^k)) }
#      Match(S, x) = True  jika x = S[i…i+|x|] untuk suatu i
#
#    Implementasi:
#    - Boleh mengambil kata dari posisi TENGAH atau AKHIR judul.
#    - Boleh mengambil substring internal suatu kata (bukan hanya awalan).
#    - Contoh: "learning" → "earn", "lear", "rning"
#    - Dipilih 1–2 token dari posisi bukan-awal, sebagian token
#      bisa dipotong menjadi substring internal.
#
#    Contoh:
#      "deep learning with python"  → "earn pyt"
#      "the art of war"             → "ar"
#      "introduction to algorithms" → "oduction algo"
# =================================================================
# def generate_mode3_partial(title: str) -> str:
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

def generate_mode3_partial(title: str) -> str:
    """
    Model LR untuk Mode 3 — PATTERN MATCH.
    Mengambil prefix token dari posisi non-awal judul, menghasilkan
    query natural yang cukup spesifik namun realistis sebagai input user.
 
    Sesuai Krishnan et al. (2017) Eq. 5 dengan modifikasi implementasi:
      PatternMatch = { Ti | Ti ∈ T ∧ (∧ P^k ∈ P* Match(Ti, P^k)) }
      Match(S, x)  = True jika x = S[i…i+|x|] untuk suatu i
 
    Aturan:
    - Token diambil dari posisi index >= 1 di judul (non-awal).
    - Hanya token NON-STOP WORD dengan len >= 5.
    - 85% kasus: prefix token (potong 50–80% panjang token).
    - 15% kasus: buang 1–2 char pertama token (simulasi suffix ringan).
    - Panjang minimum substring: max(5, ceil(50% × len(token))).
    - Diutamakan 2 token; urutan token output diacak.
    """
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
    # (terjadi pada judul di mana kata bermakna hanya di posisi pertama)
    if not non_stop_nonfirst:
        non_stop_nonfirst = [
            w for w in words
            if w.lower() not in STOP_WORDS and len(w) >= 5
        ]
 
    # Tidak ada token valid → gagal
    if not non_stop_nonfirst:
        return ""
 
    # Urutkan terpanjang → terpendek; bobot seleksi = panjang token
    # sehingga token panjang lebih sering dipilih (lebih spesifik)
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
 
    def make_substring(token: str) -> str:
        """
        Bentuk 1 substring dari token:
        - 85%: prefix token, panjang 50–80% token (natural, mudah dikenali)
        - 15%: buang 1–2 char pertama (simulasi suffix ringan)
        - Panjang minimum: max(5, ceil(50% × len(token)))
        """
        min_len = max(5, math.ceil(len(token) * 0.50))
        if len(token) < min_len:
            return ""
 
        if random.random() < 0.85:
            # Prefix token: ambil 50–80% panjang token dari awal
            max_cut = max(min_len, math.ceil(len(token) * 0.80))
            cut = random.randint(min_len, min(max_cut, len(token)))
            return token[:cut]
        else:
            # Suffix ringan: buang 1–2 char pertama saja
            max_skip = min(2, len(token) - min_len)
            if max_skip < 1:          # token pas di batas min_len, tidak bisa skip
                return token[:min_len]
            skip = random.randint(1, max_skip)
            sub  = token[skip:]
            return sub if len(sub) >= min_len else token[:min_len]
 
    result_tokens = []
    for token in picked:
        sub = make_substring(token)
        if sub:
            result_tokens.append(sub)
 
    # Acak urutan token output (Mode 3 bebas posisi, tidak harus urut judul)
    random.shuffle(result_tokens)
 
    query = " ".join(result_tokens).strip()
    return query


# =================================================================
# 5. MODE 4 — RELAXED PATTERN MATCH  (Krishnan et al., 2017 — Eq. 7)
#    Model GEN (sebagian): simulasi kesalahan ketik berbasis
#    keyboard QWERTY adjacency (Gaussian distribution, σ=0.19).
#    P_append = 0.80, P_delete = 0.04 (Section 5)
#
#    Definisi formal (paper, Section 3):
#      RelaxedPatternMatch_δ = { Ti | Ti ∈ T ∧
#        (∧ P^k ∈ P* RelaxedMatch(Ti, P^k, δ)) }
#      RelaxedMatch(S, x, δ) = True jika x dan S[i…i+|x|]
#        berbeda paling banyak δ karakter untuk suatu i
#
#    Empat jenis kesalahan ketik (Damerau-Levenshtein):
#    a. Substitution  : karakter diganti dengan tetangga QWERTY
#                       (prioritas utama, mengacu keyboard adjacency)
#    b. Deletion      : karakter dihapus
#    c. Insertion     : karakter ekstra disisipkan (tetangga QWERTY)
#    d. Transposition : dua karakter bertukar posisi
#
#    Distribusi probabilitas jenis error (dalam paper: substitution
#    adalah tipe dominan berbasis keyboard adjacency):
#      P(substitution)  = 0.60
#      P(deletion)      = 0.15
#      P(insertion)     = 0.15
#      P(transposition) = 0.10
#
#    Stop word TIDAK ditypo agar query masih dapat ditemukan sistem.
# =================================================================

# =================================================================
# 5. MODE 4 — RELAXED PATTERN MATCH  (Krishnan et al., 2017 — Eq. 7)
#    Model GEN (sebagian): simulasi kesalahan ketik berbasis
#    keyboard QWERTY adjacency (Gaussian distribution, σ=0.19).
#    P_append=0.80, P_delete=0.04, Jump=0.16 (Section 5, model GEN).
#
#    Definisi formal (paper, Section 3, Eq. 6–7):
#      RelaxedMatch(S, x, δ) = True  jika x dan S[i…i+|x|]
#        berbeda paling banyak δ karakter untuk suatu i
#      RelaxedPatternMatch_δ = { Ti | Ti ∈ T ∧
#        (∧ P^k ∈ P* RelaxedMatch(Ti, P^k, δ)) }
#
#    Parameter δ (toleransi error) direpresentasikan sebagai level:
#      Level 1 (δ=1) : 1 karakter ditypo  → 70% dari query Mode 4
#      Level 2 (δ=2) : 2 karakter ditypo  → 25% dari query Mode 4
#      Level 3 (δ=3) : 3 karakter ditypo  →  5% dari query Mode 4
#
#    Justifikasi distribusi 70/25/5:
#      Krishnan et al. (2017) Section 5 menggunakan distribusi
#      Gaussian dengan σ=0.19 untuk memilih karakter pada model GEN,
#      yang berarti karakter target dipilih dengan probabilitas
#      tertinggi dan typo ringan jauh lebih sering terjadi daripada
#      typo berat. Distribusi 70/25/5 mencerminkan kecenderungan
#      ini: mayoritas kesalahan ketik pengguna nyata hanya melibatkan
#      satu karakter (δ=1), sebagian kecil dua karakter (δ=2), dan
#      sangat jarang tiga karakter (δ=3).
#
#    Mekanisme typo:
#      HANYA substitution berbasis keyboard adjacency QWERTY,
#      sesuai mekanisme eksplisit yang disebutkan Krishnan et al.
#      (2017) Section 5. Jenis error lain (deletion, insertion,
#      transposition) TIDAK diimplementasikan karena tidak disebut
#      secara eksplisit dalam paper referensi utama.
# =================================================================

# ─────────────────────────────────────────────────────────────────
# DISTRIBUSI TYPO LEVEL (δ) untuk Mode 4
#
# Merepresentasikan parameter toleransi error δ pada Eq. 7 paper.
# Distribusi condong ke Level 1 (typo ringan), konsisten dengan
# model Gaussian σ=0.19 di Krishnan et al. (2017) Section 5.
#
#   Level  δ   Jumlah karakter ditypo   Proporsi
#   ─────  ──  ──────────────────────   ────────
#     1    1   1 karakter               70%
#     2    2   2 karakter               25%
#     3    3   3 karakter                5%
# ─────────────────────────────────────────────────────────────────
TYPO_LEVEL_PROBS = [
    (1, 0.70),   # Level 1 (δ=1) : 1 typo — 70%
    (2, 0.25),   # Level 2 (δ=2) : 2 typo — 25%
    (3, 0.05),   # Level 3 (δ=3) : 3 typo —  5%
]


def _pick_typo_level() -> int:
    """
    Pilih jumlah karakter yang akan ditypo (1, 2, atau 3)
    berdasarkan distribusi TYPO_LEVEL_PROBS.

    Return
    ------
    int — jumlah karakter yang harus ditypo (δ)
    """
    r     = random.random()
    cumul = 0.0
    for level, prob in TYPO_LEVEL_PROBS:
        cumul += prob
        if r < cumul:
            return level
    return 1  # fallback


def _substitute_char(ch: str) -> str:
    """
    Ganti satu karakter dengan tetangga QWERTY-nya.
    Merepresentasikan Gaussian keyboard adjacency (σ=0.19)
    dari Krishnan et al. (2017) Section 5.

    Parameter
    ----------
    ch : str — satu karakter alfabet lowercase

    Return
    ------
    str — karakter pengganti (tetangga QWERTY), atau ch jika
          karakter tidak ada di peta QWERTY.
    """
    if ch in QWERTY:
        return random.choice(QWERTY[ch])
    return ch


def generate_mode4_typo(text: str) -> tuple[str, int]:
    """
    Model GEN (sebagian) untuk Mode 4 — RELAXED PATTERN MATCH.
    Menghasilkan query dengan kesalahan ketik berbasis substitusi
    keyboard adjacency QWERTY, sesuai Krishnan et al. (2017) Section 5.

    Mekanisme:
    1. Tentukan jumlah typo (δ) berdasarkan distribusi level.
    2. Kumpulkan semua posisi karakter alfabet yang bisa ditypo
       dari seluruh kata BUKAN stop word.
    3. Pilih δ posisi yang BERBEDA secara acak (tanpa pengulangan
       pada posisi yang sama — sesuai poin 6 spesifikasi).
    4. Terapkan substitusi QWERTY pada posisi-posisi tersebut.
    5. Batasi δ ke jumlah karakter alfabet yang tersedia jika
       judul terlalu pendek (sesuai poin 5 spesifikasi).

    Parameter
    ----------
    text : str — judul buku sumber

    Return
    ------
    tuple[str, int]
      [0] query_text : teks query hasil typo
      [1] typo_level : jumlah karakter yang berhasil ditypo (δ aktual)
    """
    words = text.strip().split()

    # Bangun peta: (word_idx, char_idx) → karakter alfabet yang bisa disubstitusi
    # Hanya dari kata bukan stop word dengan panjang >= 3
    eligible_positions = []
    for wi, word in enumerate(words):
        if word.lower() in STOP_WORDS or len(word) < 3:
            continue
        for ci, ch in enumerate(word):
            if ch.isalpha() and ch in QWERTY:
                eligible_positions.append((wi, ci))

    # Fallback: jika tidak ada posisi eligible, gunakan kata terpanjang
    if not eligible_positions:
        longest_wi = max(range(len(words)), key=lambda i: len(words[i]))
        for ci, ch in enumerate(words[longest_wi]):
            if ch.isalpha() and ch in QWERTY:
                eligible_positions.append((longest_wi, ci))

    # Tidak ada karakter yang bisa ditypo sama sekali
    if not eligible_positions:
        return text, 0

    # Tentukan level (δ), batasi ke jumlah posisi tersedia
    target_delta = _pick_typo_level()
    actual_delta = min(target_delta, len(eligible_positions))

    # Pilih posisi yang BERBEDA (tanpa duplikat) secara acak
    chosen_positions = set()
    sampled = random.sample(eligible_positions, actual_delta)
    for pos in sampled:
        chosen_positions.add(pos)

    # Terapkan substitusi pada posisi terpilih
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


# =================================================================
# 6. GENERATE QUERY UJI

# =================================================================
# 6. GENERATE QUERY UJI
# =================================================================
def generate_queries(df: pd.DataFrame, n_total: int,
                     seed: int = 42) -> list[dict]:
    """
    Membentuk query uji sesuai jumlah Slovin dan 5 label QAC
    (mencakup Mode 1–4 dari Krishnan et al., 2017).

    Parameter
    ----------
    df      : DataFrame dengan kolom 'title' dan 'author'
    n_total : jumlah total query (hasil Slovin)
    seed    : random seed untuk reprodusibilitas

    Kolom output setiap query:
      query_id   : ID unik (Q001, Q002, …)
      query_text : teks query yang dibentuk
      mode_qac   : label mode QAC (MODE1_PREFIX_TITLE, dst.)
      model_gen  : model pembangkit (LR atau GEN)
      typo_level : jumlah karakter yang ditypo (hanya Mode 4: 1/2/3)
                   None untuk mode lain
      title      : judul buku sumber
      author     : pengarang buku sumber
    """
    random.seed(seed)
    books = df.to_dict("records")

    # ── POPULARITY-WEIGHTED SORT ──────────────────────────────────
    # Tujuan: buku populer (numRatings tinggi) lebih sering terpilih
    # sebagai sumber query, sehingga query uji lebih representatif
    # terhadap buku yang memang banyak dicari pengguna nyata.
    #
    # Cara kerja (popularity_sort):
    #   1. Urutkan pool dari numRatings terbesar ke terkecil.
    #   2. Tambahkan sedikit noise acak (0–30% dari posisi pool)
    #      agar hasilnya tidak selalu deterministik 100% — masih ada
    #      buku rating sedang yang bisa terpilih (menghindari bias
    #      ekstrem ke buku blockbuster saja).
    #   3. Urutan akhir = sort berdasarkan (rank + noise).
    #
    # Contoh ilustrasi (pool 5 buku):
    #   Harry Potter  numRatings=8.000.000 → rank 0 + noise kecil → posisi depan
    #   Dune          numRatings=1.200.000 → rank 1 + noise → posisi 1-2
    #   Buku obscure  numRatings=500       → rank 4 + noise → posisi belakang
    #
    # Dengan cara ini, query lebih banyak berasal dari buku terkenal,
    # tapi tetap ada variasi agar dataset uji tidak monoton.
    def popularity_sort(pool: list) -> list:
        """Urutkan pool berdasarkan numRatings + noise acak kecil."""
        n = len(pool)
        # Urutkan dulu dari populer ke tidak populer
        sorted_pool = sorted(pool, key=lambda b: b.get("numRatings", 0), reverse=True)
        # Tambah noise: setiap buku mendapat offset acak 0 s/d (30% n)
        noise_range = max(1, int(n * 0.30))
        scored = [(i + random.randint(0, noise_range), b)
                  for i, b in enumerate(sorted_pool)]
        scored.sort(key=lambda x: x[0])
        return [b for _, b in scored]

    # Hitung jumlah per label
    n_pt  = math.ceil(n_total * PROPORSI["MODE1_PREFIX_TITLE"])
    n_pa  = math.ceil(n_total * PROPORSI["MODE1_PREFIX_AUTHOR"])
    n_m2  = math.ceil(n_total * PROPORSI["MODE2_MULTI_TERM_PREFIX"])
    n_m3  = math.ceil(n_total * PROPORSI["MODE3_PARTIAL"])
    n_m4  = n_total - n_pt - n_pa - n_m2 - n_m3   # sisa ke Mode 4

    print(f"\n{'='*62}")
    print(f"  DISTRIBUSI QUERY (Total Slovin = {n_total})")
    print(f"  Referensi: Krishnan et al. (2017) Mode 1–4")
    print(f"{'='*62}")
    print(f"  MODE1_PREFIX_TITLE      (Mode 1, LR ) : {n_pt:3d} query (20%)")
    print(f"  MODE1_PREFIX_AUTHOR     (Mode 1, LR ) : {n_pa:3d} query (20%)")
    print(f"  MODE2_MULTI_TERM_PREFIX (Mode 2, LR ) : {n_m2:3d} query (20%)")
    print(f"  MODE3_PARTIAL           (Mode 3, LR ) : {n_m3:3d} query (20%)")
    print(f"  MODE4_TYPO              (Mode 4, GEN) : {n_m4:3d} query (20%)")
    print(f"  Proporsi seragam antar Mode 1-4 (balanced experimental design)")
    print(f"  Typo: substitusi QWERTY saja, level 70/25/5 (delta=1/2/3)")
    print(f"{'='*62}\n")

    queries    = []
    used_texts = set()   # cegah duplikat

    def try_add(query_text: str, mode_qac: str,
                model_gen: str, book: dict,
                typo_level: int | None = None) -> bool:
        """Tambahkan query jika belum duplikat dan tidak kosong."""
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

    # ── MODE 1: PREFIX TITLE ──────────────────────────────────────
    # Filter: judul >= 3 kata DAN harus punya >= 1 kata konten valid
    # (non-stop word, len >= 5) agar prefix yang dihasilkan tidak
    # hanya berisi stop word atau kata terlalu umum/pendek.
    def has_content_word(title_str: str) -> bool:
        return any(w not in STOP_WORDS and len(w) >= 5
                   for w in title_str.split())

    pool = [b for b in books
            if isinstance(b["title"], str)
            and len(b["title"].split()) >= 3
            and has_content_word(b["title"])]
    pool = popularity_sort(pool)
    count = 0
    for book in pool:
        if count >= n_pt:
            break
        q = generate_mode1_prefix_title(book["title"])
        if try_add(q, "MODE1_PREFIX_TITLE", "LR", book):
            count += 1
    print(f"  OK MODE1_PREFIX_TITLE      : {count} query terbentuk")

    # ── MODE 1: PREFIX AUTHOR ─────────────────────────────────────
    # Filter: author >= 2 kata, tidak ada inisial tunggal (< 3 char).
    pool = [b for b in books
            if isinstance(b.get("author"), str)
            and len(b["author"].strip().split()) >= 2
            and all(len(w) >= 3 for w in b["author"].strip().split())]
    pool = popularity_sort(pool)
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

    # ── MODE 2: MULTI-TERM PREFIX ─────────────────────────────────
    # Filter: judul >= 3 kata agar ada >= 2 token non-stop word
    # yang bisa dijadikan multi-term prefix.
    pool = [b for b in books
            if isinstance(b["title"], str)
            and len(b["title"].split()) >= 3]
    pool = popularity_sort(pool)
    count = 0
    for book in pool:
        if count >= n_m2:
            break
        q = generate_mode2_multi_term_prefix(book["title"])
        if try_add(q, "MODE2_MULTI_TERM_PREFIX", "LR", book):
            count += 1
    print(f"  OK MODE2_MULTI_TERM_PREFIX : {count} query terbentuk")

    # ── MODE 3: PATTERN MATCH ─────────────────────────────────────
    # Filter ketat: judul harus punya >= 2 token NON-STOP WORD
    # dengan panjang >= 5 karakter, agar fungsi generate bisa
    # menghasilkan 2-token query yang spesifik dan tidak kosong.
    # Judul pendek seperti "the name of the star" (token non-stop
    # hanya "name"(4), "star"(4)) dieksklusi dari pool ini.
    def has_enough_long_tokens(title_str: str) -> bool:
        toks = [w for w in title_str.split()
                if w.lower() not in STOP_WORDS and len(w) >= 5]
        return len(toks) >= 2
 
    pool = [b for b in books
            if isinstance(b["title"], str)
            and len(b["title"].split()) >= 4
            and has_enough_long_tokens(b["title"])]
    pool = popularity_sort(pool)
    count = 0
    for book in pool:
        if count >= n_m3:
            break
        q = generate_mode3_partial(book["title"])
        if try_add(q, "MODE3_PARTIAL", "LR", book):
            count += 1
    print(f"  OK MODE3_PARTIAL           : {count} query terbentuk")

    # ── MODE 4: RELAXED PATTERN MATCH (TYPO) ─────────────────────
    # Filter: judul >= 2 kata agar ada setidaknya 1 kata konten
    # yang bisa ditypo selain stop word.
    # Setiap query diberi metadata typo_level (δ aktual) sesuai
    # Eq. 7 Krishnan et al. (2017) — jumlah karakter yang ditypo.
    # Distribusi level: 70% Level-1, 25% Level-2, 5% Level-3.
    pool = [b for b in books
            if isinstance(b["title"], str)
            and len(b["title"].split()) >= 2]
    pool = popularity_sort(pool)
    count        = 0
    level_counts = {1: 0, 2: 0, 3: 0}   # statistik level typo
    for book in pool:
        if count >= n_m4:
            break
        q, delta = generate_mode4_typo(book["title"])
        # Pastikan hasil typo benar-benar berbeda dari aslinya
        if delta > 0 and q.lower() != book["title"].lower():
            if try_add(q, "MODE4_TYPO", "GEN", book, typo_level=delta):
                count += 1
                level_counts[min(delta, 3)] += 1
    print(f"  OK MODE4_TYPO              : {count} query terbentuk")
    print(f"     Level 1 (delta=1): {level_counts[1]} query")
    print(f"     Level 2 (delta=2): {level_counts[2]} query")
    print(f"     Level 3 (delta=3): {level_counts[3]} query")

    return queries


# =================================================================
# 7. LOGGING & SIMPAN
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


def print_statistics(queries: list):
    from collections import Counter
    modes = Counter(q["mode_qac"] for q in queries)
    total = len(queries)

    print(f"\n{'='*62}")
    print(f"  STATISTIK QUERY UJI")
    print(f"{'='*62}")
    print(f"  Total : {total} queries\n")
    print("  MODE QAC:")
    labels = [
        "MODE1_PREFIX_TITLE",
        "MODE1_PREFIX_AUTHOR",
        "MODE2_MULTI_TERM_PREFIX",
        "MODE3_PARTIAL",
        "MODE4_TYPO",
    ]
    for label in labels:
        c   = modes[label]
        pct = c / total * 100 if total else 0
        bar = "X" * int(pct / 5)
        print(f"    {label:26} {c:4}  ({pct:5.1f}%)  {bar}")

    print(f"\n  MODEL PEMBANGKIT:")
    models = Counter(q["model_gen"] for q in queries)
    for m, c in models.items():
        pct = c / total * 100 if total else 0
        print(f"    {m:6} : {c:4} query ({pct:.1f}%)")

    typo_queries = [q for q in queries if q["mode_qac"] == "MODE4_TYPO"]
    if typo_queries:
        lvl_counts = Counter(q.get("typo_level") for q in typo_queries)
        n_typo     = len(typo_queries)
        print(f"\n  BREAKDOWN TYPO LEVEL (delta) — Mode 4:")
        print(f"  Distribusi target: 70% Level-1 / 25% Level-2 / 5% Level-3")
        print(f"  (Konsisten dengan Gaussian sigma=0.19, Krishnan et al. 2017 Sec.5)")
        for lvl in sorted(lvl_counts):
            c   = lvl_counts[lvl]
            pct = c / n_typo * 100
            bar = "X" * int(pct / 5)
            print(f"    Level {lvl} (delta={lvl}) : {c:4} query ({pct:5.1f}%)  {bar}")


# =================================================================
# MAIN
# =================================================================
def main():
    # Reset log
    with open(LOG_FILE, "w") as f:
        f.write("")

    log("=" * 62)
    log("START: Pembentukan Query Uji (QAC Taxonomy Mode 1-4)")
    log("Referensi: Krishnan et al. (ADCS 2017)")
    log("=" * 62)

    # -- Load CSV --------------------------------------------------
    log(f"Loading CSV: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE, usecols=["title", "author", "numRatings"])
    df = df.dropna(subset=["title"])
    df["title"]      = df["title"].str.lower().str.strip()
    df["author"]     = df["author"].fillna("").str.lower().str.strip()
    # numRatings: isi NaN dengan 0, pastikan integer
    df["numRatings"] = pd.to_numeric(df["numRatings"], errors="coerce").fillna(0).astype(int)
    df = df.drop_duplicates(subset=["title"])
    df = df.reset_index(drop=True)
    N  = len(df)
    log(f"Dataset: {N:,} records unik")
    log(f"numRatings — min={df['numRatings'].min():,}  "
        f"max={df['numRatings'].max():,}  "
        f"median={int(df['numRatings'].median()):,}")

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

    log("=" * 62)
    log(f"DONE! {len(queries)} query uji tersimpan -> {OUTPUT_JSON}")
    log("=" * 62)


if __name__ == "__main__":
    main()