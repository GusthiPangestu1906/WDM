import os
import pandas as pd

# Load dataset pengujian
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, '..', 'data', 'titanic_test.csv')
if not os.path.exists(csv_path):
    csv_path = 'data/titanic_test.csv'

test_dataset = pd.read_csv(csv_path)

print("=== 2. Load Dataset Test (titanic_test.csv) ===")
print(f"File Path : {csv_path}")
print(f"Dimensi   : {test_dataset.shape[0]} baris, {test_dataset.shape[1]} kolom\n")

# Tampilkan sampel data secara acak (random sample)
print("--- Sampel 5 Baris Data Acak (Random Sample) ---")
print(test_dataset.sample(5, random_state=42))

print("\n--- Ringkasan Kolom ---")
print(test_dataset.info())
