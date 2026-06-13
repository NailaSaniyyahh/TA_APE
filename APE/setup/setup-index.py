from opensearchpy import OpenSearch
import pandas as pd
import os
import math

client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_auth=("admin", "KoTA404TABAH!"),
    use_ssl=True,
    verify_certs=False,
    ssl_show_warn=False,
    timeout=60
)

INDEX_NAME = "books2"

mapping = {
    "mappings": {
        "properties": {
            "title": {"type": "text"},
            "author": {"type": "text"},
            "description": {"type": "text"},
            "numRatings": {
                "type": "integer"         # jumlah rating mentah
            },
            "suggest_title": {
                "type": "completion",   # suggest khusus dari judul
                "preserve_separators": False,
                "preserve_position_increments": False
            },
            "suggest_author": {
                "type": "completion",   # suggest khusus dari author
                "preserve_separators": False,
                "preserve_position_increments": False
            }
        }
    }
}

if client.indices.exists(index=INDEX_NAME):
    client.indices.delete(index=INDEX_NAME)
    print(f"Index '{INDEX_NAME}' lama dihapus.")

client.indices.create(index=INDEX_NAME, body=mapping)
print(f"Index '{INDEX_NAME}' berhasil dibuat.")

csv_path = os.path.join("..", "..", "..", "DATASET", "hasil_books2.csv")
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip().str.lower()
df = df.fillna("")

if "numratings" not in df.columns:
    print("⚠ WARNING: kolom numRatings tidak ditemukan!")
    df["numratings"] = 0

# normalisasi num rating
max_ratings = pd.to_numeric(df["numratings"], errors="coerce").max()
max_ratings = max_ratings if max_ratings > 0 else 1
log_max = math.log1p(float(max_ratings))

print(f"✓ Max numRatings: {max_ratings:,}")
print(f"✓ Loaded {len(df):,} buku dari {csv_path}")

def make_suggest_inputs(title: str) -> list:
    tokens = title.lower().split()
    inputs = [title.lower()]
    for i in range(1, min(len(tokens), 4)):   # max 3 variasi tambahan
        inputs.append(" ".join(tokens[i:]))
    return list(dict.fromkeys(inputs))      

success = 0
failed  = 0

for _, row in df.iterrows():
    title      = str(row.get("title", "")).strip()
    author     = str(row.get("author", "")).strip()
    description = str(row.get("description", "")).strip()
    numRatings = int(row.get("numratings", 0) or 0)

    if not title:
        failed += 1
        continue

    log_ratings = math.log1p(float(numRatings))
    weight      = max(1, int((log_ratings / log_max) * 1000))

    doc = {
        "title": title,
        "author": author,
        "description": description,
        "numRatings" : numRatings,
        "suggest_title": {
            "input": make_suggest_inputs(title),
            "weight": weight
        },
        "suggest_author": {
            "input": [author.lower()] if author else [],
            "weight": weight
        }
    }

    try:
        client.index(index=INDEX_NAME, body=doc)
        success += 1
    except Exception as e:
        print(f"  ✗ Gagal index '{title}': {e}")
        failed += 1

    if success % 1000 == 0:
        print(f"  Progress: {success:,} buku terindeks...")

print(f"\n✅ Indexing selesai!")
print(f"   Berhasil : {success:,} buku")
print(f"   Gagal    : {failed:,} buku")