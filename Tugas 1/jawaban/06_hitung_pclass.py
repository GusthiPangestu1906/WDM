import os
import pandas as pd

# Load dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, '..', 'data', 'titanic.csv')
if not os.path.exists(csv_path):
    csv_path = 'data/titanic.csv'

dataset = pd.read_csv(csv_path)

# Hitung jumlah penumpang tiap kelas
pclass_counts = dataset['Pclass'].value_counts().sort_index()

print("=== Jumlah Penumpang per Pclass ===")
for pclass, count in pclass_counts.items():
    print(f"Pclass {pclass} : {count} penumpang")
