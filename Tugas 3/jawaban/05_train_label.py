import os
import pandas as pd

# Load dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, '..', 'data', 'titanic.csv')
if not os.path.exists(csv_path):
    csv_path = 'data/titanic.csv'

dataset = pd.read_csv(csv_path)

# Identifikasi baris missing value pada fitur (Age, Fare)
train_features = dataset[["Age", "Fare"]]
pos_missing_train = train_features.isna().any(axis=1)

# Ambil label target Survived hanya untuk baris yang bersih
train_label = dataset.loc[~pos_missing_train, "Survived"]

print("=== 5. Label Train (Survived) ===")
print(f"Jumlah label train (valid) : {len(train_label)}")
print(f"\nDistribusi Kelas:\n{train_label.value_counts()}")

# Tampilkan sampel data secara acak (random sample)
print("\n--- Sampel 10 Label Secara Acak (Random Sample) ---")
print(train_label.sample(10, random_state=42))
