import os
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, '..', 'data', 'titanic.csv')
if not os.path.exists(csv_path):
    csv_path = 'data/titanic.csv'

dataset = pd.read_csv(csv_path)

print("=== 1. Load Dataset Train (titanic.csv) ===")
print(f"File Path : {csv_path}")
print(f"Dimensi   : {dataset.shape[0]} baris, {dataset.shape[1]} kolom\n")

print("--- Sampel 5 Baris Data Acak (Random Sample) ---")
print(dataset.sample(5, random_state=42))

print("\n--- Ringkasan Kolom ---")
print(dataset.info())