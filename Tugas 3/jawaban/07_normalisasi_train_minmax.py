import os
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, '..', 'data', 'titanic.csv')
if not os.path.exists(csv_path):
    csv_path = 'data/titanic.csv'

dataset = pd.read_csv(csv_path)

# Ekstraksi fitur train & hapus missing value
train_features = dataset[["Age", "Fare"]]
pos_missing_train = train_features.isna().any(axis=1)
train_data = train_features[~pos_missing_train].copy()

# 7. Normalisasi Min-Max (0 - 1) & catat min dan max
min_val = train_data.min()
max_val = train_data.max()

norm_train_data = (train_data - min_val) / (max_val - min_val)

print("=== 7. Normalisasi Min-Max 0-1 pada train_data ===")
print("Nilai Minimum Atribut Train:")
print(min_val)
print("\nNilai Maksimum Atribut Train:")
print(max_val)

# Ambil sampel acak menggunakan index yang sama untuk membandingkan sebelum vs sesudah
sample_indices = train_data.sample(5, random_state=42).index

print("\n--- Sampel 5 Baris train_data Sebelum Normalisasi (Acak) ---")
print(train_data.loc[sample_indices])

print("\n--- Sampel 5 Baris train_data Setelah Normalisasi (Acak) ---")
print(norm_train_data.loc[sample_indices])
