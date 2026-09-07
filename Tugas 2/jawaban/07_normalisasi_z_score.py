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

# Normalisasi Z-Score
mean_val = data.mean()
std_val = data.std()
norm_zscore = (data - mean_val) / std_val

# Tampilkan hasil
print("=== Normalisasi Z-Score ===")
print(norm_zscore)
