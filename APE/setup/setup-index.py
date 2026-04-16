from opensearchpy import OpenSearch
import pandas as pd
import os

# Koneksi ke OpenSearch
client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_auth=("admin", "KoTA404TABAH!"),
    use_ssl=True,
    verify_certs=False,
    ssl_show_warn=False,
    timeout=60
)

INDEX_NAME = "books"

# Mapping index dengan completion suggester
mapping = {
    "mappings": {
        "properties": {
            "title": {"type": "text"},
            "author": {"type": "text"},
            "description": {"type": "text"},
            "suggest_title": {
                "type": "completion"   # suggest khusus dari judul
            },
            "suggest_author": {
                "type": "completion"   # suggest khusus dari author
            }
        }
    }
}

# Hapus index lama kalau ada, buat ulang
if client.indices.exists(index=INDEX_NAME):
    client.indices.delete(index=INDEX_NAME)
    print(f"Index '{INDEX_NAME}' lama dihapus.")

client.indices.create(index=INDEX_NAME, body=mapping)
print(f"Index '{INDEX_NAME}' berhasil dibuat.")

# Baca CSV
csv_path = os.path.join("..", "..", "..", "DATASET", "books_clean_en.csv")
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip().str.lower()
df = df.fillna("")

# Index tiap buku
success = 0
for _, row in df.iterrows():
    title = row.get("title", "")
    author = row.get("author", "")
    description = row.get("description", "")

    if not title:
        continue

    doc = {
        "title": title,
        "author": author,
        "description": description,
        "suggest_title": {
            "input": [title]          # HANYA judul
        },
        "suggest_author": {
            "input": [author] if author else []   # HANYA author
        }
    }

    client.index(index=INDEX_NAME, body=doc)
    success += 1

print(f"Berhasil index {success} buku.")