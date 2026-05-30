import pandas as pd

input_path = "../hasil_books_en.csv"
# books_clean_en
# hasil_books_en
df = pd.read_csv(input_path)

df['title_len'] = df['title'].apply(lambda x: len(str(x).split()))
df['author_len'] = df['author'].apply(lambda x: len(str(x).split()))
df['description_len'] = df['description'].apply(lambda x: len(str(x).split()))

print(f"Total akhir (English): {len(df)} baris")

print("Panjang kata di title:", df['title_len'].mean())
print("Panjang kata di author:", df['author_len'].mean())
print("Panjang kata di description:", df['description_len'].mean())

print("Panjang minimun kata di title:", df['title_len'].min())
print("Panjang minimum kata di author:", df['author_len'].min())
print("Panjang minimum kata di description:", df['description_len'].min())

print("Panjang maksimum kata di title:", df['title_len'].max())
print("Panjang maksimum kata di author:", df['author_len'].max())
print("Panjang maksimum kata di description:", df['description_len'].max())