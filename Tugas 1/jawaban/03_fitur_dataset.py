import os
import pandas as pd

# Load dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, '..', 'data', 'titanic.csv')
if not os.path.exists(csv_path):
    csv_path = 'data/titanic.csv'

dataset = pd.read_csv(csv_path)

# Ambil kolom-kolom fitur yang ditentukan
data = dataset[['Name', 'Sex', 'Age', 'Pclass', 'Fare']]

# Tampilkan hasil
print("=== Kolom Fitur (Name, Sex, Age, Pclass, Fare) ===")
print(data)
