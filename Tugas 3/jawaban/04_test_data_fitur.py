import os
import pandas as pd

# Load test dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, '..', 'data', 'titanic_test.csv')
if not os.path.exists(csv_path):
    csv_path = 'data/titanic_test.csv'

test_dataset = pd.read_csv(csv_path)

# Ambil fitur (Age, Fare)
test_features = test_dataset[["Age", "Fare"]]

# Catat posisi baris yang memiliki missing value
pos_missing_test = test_features.isna().any(axis=1)

# Hapus baris yang memiliki missing value
test_data = test_features[~pos_missing_test].copy()

print("=== 4. Fitur Test Data (Age, Fare) & Pembersihan Missing Value ===")
print(f"Jumlah baris awal              : {len(test_features)}")
print(f"Jumlah baris missing value     : {pos_missing_test.sum()}")
print(f"Jumlah baris bersih (test_data): {len(test_data)}\n")

# Sampel indeks baris missing value secara acak
print("--- Sampel Indeks Baris yang Memiliki Missing Value (Acak) ---")
sample_missing_idx = test_features[pos_missing_test].sample(10, random_state=42).index.tolist()
print(sample_missing_idx)

# Sampel data bersih secara acak
print("\n--- Sampel 5 Baris test_data Secara Acak (Random Sample) ---")
print(test_data.sample(5, random_state=42))
