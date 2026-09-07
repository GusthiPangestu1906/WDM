import os
import pandas as pd

# Load dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, '..', 'data', 'titanic.csv')
if not os.path.exists(csv_path):
    csv_path = 'data/titanic.csv'

dataset = pd.read_csv(csv_path)

# Tampilkan seluruh data
print("=== Data Titanic ===")
print(dataset)

# Tampilkan 5 baris teratas
print("\n=== 5 Baris Pertama ===")
print(dataset.head())
