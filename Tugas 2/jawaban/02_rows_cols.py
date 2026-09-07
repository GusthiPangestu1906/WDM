import os
import pandas as pd

# Load dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, '..', 'data', 'titanic.csv')
if not os.path.exists(csv_path):
    csv_path = 'data/titanic.csv'

dataset = pd.read_csv(csv_path)

# Ambil jumlah baris dan kolom
rows, cols = dataset.shape

print(f"Jumlah Baris : {rows}")
print(f"Jumlah Kolom : {cols}")
