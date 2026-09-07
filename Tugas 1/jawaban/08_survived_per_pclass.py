import os
import pandas as pd

# Load dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, '..', 'data', 'titanic.csv')
if not os.path.exists(csv_path):
    csv_path = 'data/titanic.csv'

dataset = pd.read_csv(csv_path)

# Rekap jumlah selamat & tidak selamat per Pclass
rekap = pd.crosstab(
    dataset['Pclass'], 
    dataset['Survived'], 
    rownames=['Pclass'], 
    colnames=['Survived']
)
rekap.rename(columns={0: 'Tidak Selamat (0)', 1: 'Selamat (1)'}, inplace=True)
rekap['Total'] = rekap.sum(axis=1)

print("=== Rekap Survived per Pclass ===")
print(rekap)
