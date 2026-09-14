import os
import pandas as pd

# Load dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, '..', 'data', 'titanic.csv')
if not os.path.exists(csv_path):
    csv_path = 'data/titanic.csv'

dataset = pd.read_csv(csv_path)

# Ambil fitur (Age, Fare)
train_features = dataset[["Age", "Fare"]]

# Catat posisi baris yang memiliki missing value
pos_missing_train = train_features.isna().any(axis=1)

# Hapus baris yang memiliki missing value
train_data = train_features[~pos_missing_train].copy()

print("=== 3. Fitur Train Data (Age, Fare) & Pembersihan Missing Value ===")
print(f"Jumlah baris awal               : {len(train_features)}")
print(f"Jumlah baris missing value      : {pos_missing_train.sum()}")
print(f"Jumlah baris bersih (train_data): {len(train_data)}\n")

# Sampel indeks baris missing value secara acak
print("--- Sampel Indeks Baris yang Memiliki Missing Value (Acak) ---")
sample_missing_idx = train_features[pos_missing_train].sample(10, random_state=42).index.tolist()
print(sample_missing_idx)

# Sampel data bersih secara acak
print("\n--- Sampel 5 Baris train_data Secara Acak (Random Sample) ---")
print(train_data.sample(5, random_state=42))
