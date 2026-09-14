import os
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', 'data')

train_path = os.path.join(data_dir, 'titanic.csv')
test_path = os.path.join(data_dir, 'titanic_test.csv')

if not os.path.exists(train_path):
    train_path = 'data/titanic.csv'
if not os.path.exists(test_path):
    test_path = 'data/titanic_test.csv'

# Siapkan train_data untuk memperoleh min & max (Langkah 7)
dataset = pd.read_csv(train_path)
train_features = dataset[["Age", "Fare"]]
pos_missing_train = train_features.isna().any(axis=1)
train_data = train_features[~pos_missing_train].copy()

min_val = train_data.min()
max_val = train_data.max()

# Siapkan test_data
test_dataset = pd.read_csv(test_path)
test_features = test_dataset[["Age", "Fare"]]
pos_missing_test = test_features.isna().any(axis=1)
test_data = test_features[~pos_missing_test].copy()

# 8. Normalisasi Min-Max test_data menggunakan min & max Langkah 7 (dari train_data)
norm_test_data = (test_data - min_val) / (max_val - min_val)

print("=== 8. Normalisasi Min-Max 0-1 pada test_data (Menggunakan Min & Max Train) ===")
print("Nilai Min Acuan (dari Train):")
print(min_val)
print("\nNilai Max Acuan (dari Train):")
print(max_val)

# Ambil sampel acak menggunakan index yang sama untuk membandingkan sebelum vs sesudah
sample_indices = test_data.sample(5, random_state=42).index

print("\n--- Sampel 5 Baris test_data Sebelum Normalisasi (Acak) ---")
print(test_data.loc[sample_indices])

print("\n--- Sampel 5 Baris test_data Setelah Normalisasi (Acak) ---")
print(norm_test_data.loc[sample_indices])
