import pandas as pd
import json
import random
import math
import os
from collections import defaultdict
from datetime import datetime

# Define base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_DIR = os.path.join(BASE_DIR, 'DATASET')

CSV_FILE = os.path.join(DATASET_DIR, 'books_clean.csv')
OUTPUT_FILE = os.path.join(DATASET_DIR, 'test_queries_50.json')
LOG_FILE = os.path.join(DATASET_DIR, 'test_queries_generation.log')

TOTAL_DATASET = 52422
PHASE1_TEST_SIZE = 50


def log_message(msg, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] [{level}] {msg}"
    print(log_line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_line + '\n')

def load_csv(filepath):
    log_message(f"Loading CSV from: {filepath}")
    
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower()
    
    required_cols = ['title', 'author', 'description', 'genres']
    for col in required_cols:
        if col not in df.columns:
            log_message(f"✗ Column '{col}' not found! Columns available: {list(df.columns)}", "ERROR")
            raise ValueError(f"Missing column: {col}")
    
    df['product_id'] = df.index + 1
    df = df[df['title'].notna() & (df['title'].str.strip() != '')]
    df = df.reset_index(drop=True)
    
    log_message(f"✓ Loaded {len(df):,} valid books")
    return df

def create_test_set(df, num_queries=50):
    test_set = []
    books = df.to_dict('records')

    log_message("Generating EXACT_TITLE queries (33%)...")
    valid_title = [b for b in books if isinstance(b['title'], str) and b['title'].strip()]
    n_title = int(num_queries * 0.33)
    sample_title = random.sample(valid_title, min(n_title, len(valid_title)))

    for book in sample_title:
        test_set.append({
            "query_id": f"Q{len(test_set)+1:03d}",
            "query_text": book['title'].strip(),
            "query_type": "EXACT_TITLE",
            "query_difficulty": "EASY",
            "ground_truth": {
                "product_id": book['product_id'],
                "title": book['title'],
                "author": book['author'] if pd.notna(book['author']) else "Unknown",
                "genres": book['genres'] if pd.notna(book['genres']) else ""
            }
        })
    log_message(f"✓ {len(sample_title)} EXACT_TITLE queries")

    log_message("Generating AUTHOR queries (33%)...")
    valid_author = [b for b in books if isinstance(b['author'], str) and b['author'].strip()]
    n_author = int(num_queries * 0.33)
    sample_author = random.sample(valid_author, min(n_author, len(valid_author)))

    for book in sample_author:
        test_set.append({
            "query_id": f"Q{len(test_set)+1:03d}",
            "query_text": book['author'].strip(),
            "query_type": "AUTHOR",
            "query_difficulty": "MEDIUM",
            "ground_truth": {
                "product_id": book['product_id'],
                "title": book['title'],
                "author": book['author'],
                "genres": book['genres'] if pd.notna(book['genres']) else ""
            }
        })
    log_message(f"✓ {len(sample_author)} AUTHOR queries")

    log_message("Generating GENRE queries (34%)...")
    valid_genre = [b for b in books if isinstance(b['genres'], str) and b['genres'].strip()]
    n_genre = num_queries - len(test_set)
    sample_genre = random.sample(valid_genre, min(n_genre, len(valid_genre)))

    for book in sample_genre:
        genres_list = book['genres'].strip().split()
        keyword = " ".join(genres_list[:random.choice([1, 2])])

        test_set.append({
            "query_id": f"Q{len(test_set)+1:03d}",
            "query_text": keyword,
            "query_type": "GENRE",
            "query_difficulty": "HARD",
            "ground_truth": {
                "product_id": book['product_id'],
                "title": book['title'],
                "author": book['author'] if pd.notna(book['author']) else "Unknown",
                "genres": book['genres']
            }
        })
    log_message(f"✓ {len(sample_genre)} GENRE queries")

    return test_set

def validate_test_set(test_set):
    log_message("Validating test set...")
    texts = [q['query_text'] for q in test_set]
    dupes = len(texts) - len(set(texts))
    if dupes:
        log_message(f"⚠ {dupes} duplicate queries found", "WARNING")
    log_message(f"✓ Validation done — {len(test_set)} queries total")

def print_statistics(test_set):
    type_counts = defaultdict(int)
    diff_counts = defaultdict(int)
    for q in test_set:
        type_counts[q['query_type']] += 1
        diff_counts[q['query_difficulty']] += 1

    print(f"""
╔════════════════════════════════════════════════════════════╗
║                  TEST SET STATISTICS                       ║
╚════════════════════════════════════════════════════════════╝
Total Queries: {len(test_set)}

QUERY TYPE:""")
    for t in ['EXACT_TITLE', 'AUTHOR', 'GENRE']:
        c = type_counts[t]
        pct = c / len(test_set) * 100
        print(f"  {t:15} {c:3} ({pct:5.1f}%) {'█' * int(pct/5)}")

    print("\nDIFFICULTY:")
    for d in ['EASY', 'MEDIUM', 'HARD']:
        c = diff_counts[d]
        pct = c / len(test_set) * 100
        print(f"  {d:15} {c:3} ({pct:5.1f}%) {'█' * int(pct/5)}")

    print("\nSAMPLE (5 pertama):")
    for q in test_set[:5]:
        print(f"\n  [{q['query_id']}] \"{q['query_text']}\"")
        print(f"      Type: {q['query_type']} | Difficulty: {q['query_difficulty']}")
        print(f"      → \"{q['ground_truth']['title']}\" by {q['ground_truth']['author']}")

def save_to_json(test_set, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_set, f, indent=2, ensure_ascii=False)
    log_message(f"✓ Saved to {output_file}")

def main():
    with open(LOG_FILE, 'w') as f:
        f.write("")

    log_message("=" * 60)
    log_message("START: TEST QUERY GENERATION FROM CSV")
    log_message("=" * 60)

    try:
        df = load_csv(CSV_FILE)
        test_set = create_test_set(df, num_queries=PHASE1_TEST_SIZE)
        validate_test_set(test_set)
        save_to_json(test_set, OUTPUT_FILE)
        print_statistics(test_set)

        log_message("=" * 60)
        log_message("✓ DONE!")
        log_message(f"✓ Output: {OUTPUT_FILE}")
        log_message("=" * 60)

        return test_set

    except FileNotFoundError:
        log_message(f"✗ File '{CSV_FILE}' tidak ditemukan! Pastikan path-nya benar.", "ERROR")
    except Exception as e:
        log_message(f"✗ Error: {e}", "ERROR")
        raise

if __name__ == "__main__":
    test_set = main()
    if test_set:
        print(f"\n✅ SUCCESS! {len(test_set)} test queries generated")
        print(f"📁 Output: {OUTPUT_FILE}")
    else:
        print("\n❌ FAILED. Cek log di atas.")