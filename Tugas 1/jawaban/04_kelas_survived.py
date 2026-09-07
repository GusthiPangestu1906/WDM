import os
import pandas as pd

# Load dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, '..', 'data', 'titanic.csv')
if not os.path.exists(csv_path):
    csv_path = 'data/titanic.csv'

dataset = pd.read_csv(csv_path)

# Ambil kolom target kelas (Survived)
target_class = dataset['Survived']

# Tampilkan hasil
print("=== Kolom Kelas (Survived) ===")
print(target_class)
