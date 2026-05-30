# import pandas as pd

# # df = pd.read_csv("../books_1.Best_Books_Ever.csv")

# # print(df.head())
# # print(df.info())
# # print(df.describe())
# # print(df.isnull().sum())

# input_path = "../books_1.Best_Books_Ever.csv"
# output_path = "../books_clean.csv"

# df = pd.read_csv(input_path)

# # ngambil colom yg dimau
# # df = df[["title", "author", "description", "genres"]]
# # handle missing value
# df = df.dropna(subset=["title", "author"])
# df["description"] = df["description"].fillna("")
# df["genres"] = df["genres"].fillna("")
# # drop duplicate
# df = df.drop_duplicates()
# # normalize all columns
# text_columns = df.select_dtypes(include="object").columns
# for col in text_columns:
#     df[col] = (
#         df[col]
#         .astype(str)
#         .str.lower()
#         .str.strip()
#         .str.replace(r"[^a-z0-9\s]", "", regex=True)
#     )
# df.to_csv(output_path, index=False)

# print(f"Original rows after load: {len(pd.read_csv(input_path))}")
# print(f"Cleaned rows after processing: {len(df)}")
# print(f"Saved cleaned dataset to: {output_path}")

import pandas as pd
import re
from langdetect import detect

input_path = "../books_1.Best_Books_Ever.csv"
output_path = "../hasil_books2.csv"

df = pd.read_csv(input_path)

print("Before cleaning:", len(df))

df = df[["title", "author", "description", "numRatings"]]  # ← tambah numRatings
df = df.dropna(subset=["title", "author"])

df["title"] = df["title"].astype(str).str.strip()
df["author"] = df["author"].astype(str).str.strip()

df["description"] = df["description"].fillna("")
df["numRatings"] = pd.to_numeric(df["numRatings"], errors="coerce").fillna(0).astype(int)  # ← tambah ini

def clean_text_general(text):
    text = str(text).lower()
    # Ganti simbol [^a-z0-9\s] dengan SPASI " ", bukan ""
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Hapus spasi ganda (multiple whitespaces)
    text = " ".join(text.split())
    return text
# for col in ["title", "author", "description"]:
#     df[col] = (
#         df[col]
#         .astype(str)
#         .str.lower()
#         .str.strip()
#         .str.replace(r"[^a-z0-9\s-]", "", regex=True)
#     )

# print("After cleaning:", len(df))
# df.to_csv(output_path, index=False)

# print("Saved to:", output_path)

def clean_author(text):
    text = str(text).strip()
    if ',' in text:
        text = text.split(',')[0].strip()
    text = re.sub(r'\b(editor|narrator|illustrator|translator|commentary|author|introduction|preface|narrator|notes|anonymous)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'goodreads', '', text, flags=re.IGNORECASE)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = " ".join(text.split())  # hapus spasi ganda
    # word = text.split()
    return text


df["author"] = df["author"].apply(clean_author)
df["title"] = df["title"].apply(clean_text_general)
df["description"] = df["description"].apply(clean_text_general)


df = df[df["title"] != ""]
df = df[df["author"] != ""]

df = df.drop_duplicates(subset="title")
print(f"Setelah drop duplikat: {len(df)} baris")

def detect_lang(row):
    text = row["description"] if row["description"].strip() else row["title"]
    try:
        return detect(text)
    except:
        return 'unknown'

print("Mendeteksi bahasa...")
df["lang"] = df.apply(detect_lang, axis=1)
df = df[df["lang"] == "en"]
df = df.drop(columns=["lang"])
print(f"Setelah filter English: {len(df)} baris")

df.to_csv(output_path, index=False)
print(f"\nSelesai! Tersimpan di: {output_path}")