import os
import pandas as pd

# Load dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, '..', 'data', 'titanic.csv')
if not os.path.exists(csv_path):
    csv_path = 'data/titanic.csv'

dataset = pd.read_csv(csv_path)

# Ambil data fitur awal
data = dataset[['Name', 'Sex', 'Age', 'Pclass', 'Fare']].copy()

# Tambahkan fitur Relatives (SibSp + Parch)
data['Relatives'] = dataset['SibSp'] + dataset['Parch']

# Tampilkan hasil
print("=== Data Fitur + Relatives ===")
print(data)
