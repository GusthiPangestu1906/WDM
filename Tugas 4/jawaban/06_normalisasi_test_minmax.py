import os
import pandas as pd
from sklearn.model_selection import train_test_split

base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, '..', 'data', 'titanic.csv')
if not os.path.exists(csv_path):
    csv_path = 'data/titanic.csv'
if not os.path.exists(csv_path):
    csv_path = 'titanic.csv'

dataset = pd.read_csv(csv_path)

features = ["Sex", "Age", "Pclass", "Fare"]
target = "Survived"

# Ekstraksi fitur dan target yang dibutuhkan
data = dataset[features + [target]].copy()
data["Sex"] = data["Sex"].map({"female": 0, "male": 1})
data["Fare"] = data["Fare"].fillna(data["Fare"].median())

# Pembagian data latih (70%) dan uji (30%)
train_df, test_df = train_test_split(
    data, test_size=0.30, random_state=42, stratify=data[target]
)
train_copy = train_df.copy()
test_copy = test_df.copy()

# Imputasi missing value Age menggunakan referensi nilai mean dari data latih
mean_age_train = train_copy.groupby(target)["Age"].mean()
overall_mean_train = train_copy["Age"].mean()

for cls in mean_age_train.index:
    mask_tr = (train_copy[target] == cls) & (train_copy["Age"].isna())
    train_copy.loc[mask_tr, "Age"] = mean_age_train[cls]
train_copy["Age"] = train_copy["Age"].fillna(overall_mean_train)

for cls in mean_age_train.index:
    mask_te = (test_copy[target] == cls) & (test_copy["Age"].isna())
    test_copy.loc[mask_te, "Age"] = mean_age_train[cls]
test_copy["Age"] = test_copy["Age"].fillna(overall_mean_train)

train_data = train_copy[features].astype(float)
test_data = test_copy[features].astype(float)

# Catat nilai minimum dan maksimum dari data latih (Langkah 5)
min_val = train_data.min()
max_val = train_data.max()

# 6. Normalisasi test_data menggunakan parameter min dan max dari Langkah 5
norm_test_data = (test_data - min_val) / (max_val - min_val)

print("=== 6. Normalisasi Min-Max 0-1 pada test_data (Parameter Train) ===")
print("Nilai Acuan Minimum (dari Data Latih):")
print(min_val)
print("\nNilai Acuan Maksimum (dari Data Latih):")
print(max_val)

sample_indices = test_data.sample(5, random_state=42).index

print("\n--- Sampel 5 Baris test_data Sebelum Normalisasi (Acak) ---")
print(test_data.loc[sample_indices])

print("\n--- Sampel 5 Baris test_data Setelah Normalisasi (Acak) ---")
print(norm_test_data.loc[sample_indices])
