import os
import pandas as pd

# Load dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, '..', 'data', 'titanic.csv')
if not os.path.exists(csv_path):
    csv_path = 'data/titanic.csv'

dataset = pd.read_csv(csv_path)

# Hitung jumlah penumpang berdasarkan jenis kelamin
sex_counts = dataset['Sex'].value_counts()

print("=== Jumlah Penumpang per Sex ===")
for sex, count in sex_counts.items():
    print(f"Sex '{sex}' : {count} penumpang")
