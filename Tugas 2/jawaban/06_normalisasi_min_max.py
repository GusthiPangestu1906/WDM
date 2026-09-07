import os
import pandas as pd

# Load dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, '..', 'data', 'titanic.csv')
if not os.path.exists(csv_path):
    csv_path = 'data/titanic.csv'

dataset = pd.read_csv(csv_path)

# Pengisian missing value pada fitur Age dengan nilai mean dari masing-masing class
dataset['Age'] = dataset.groupby('Survived')['Age'].transform(
    lambda x: x.fillna(x.mean())
)

# Ambil kolom fitur (Age, Fare)
data = dataset[['Age', 'Fare']]

# Normalisasi Min-Max (0-1)
min_val = data.min()
max_val = data.max()
norm_minmax = (data - min_val) / (max_val - min_val)

# Tampilkan hasil
print("=== Normalisasi Min-Max (0-1) ===")
print(norm_minmax)
