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

# Pisahkan data latih (70%) dan uji (30%) sesuai skema validasi Hold-out
train_df, test_df = train_test_split(
    data, test_size=0.30, random_state=42, stratify=data[target]
)
train_copy = train_df.copy()

# Isi missing value Age menggunakan rata-rata usia per kelas target pada data latih
mean_age_train = train_copy.groupby(target)["Age"].mean()
overall_mean_train = train_copy["Age"].mean()

for cls in mean_age_train.index:
    mask = (train_copy[target] == cls) & (train_copy["Age"].isna())
    train_copy.loc[mask, "Age"] = mean_age_train[cls]
train_copy["Age"] = train_copy["Age"].fillna(overall_mean_train)

# Ambil data fitur pelatihan yang telah bersih
train_data = train_copy[features].astype(float)

# 5. Normalisasi Min-Max (0 - 1) & catat nilai min dan max setiap atribut
min_val = train_data.min()
max_val = train_data.max()

norm_train_data = (train_data - min_val) / (max_val - min_val)

print("=== 5. Normalisasi Min-Max 0-1 pada train_data ===")
print("Nilai Minimum Atribut Train:")
print(min_val)
print("\nNilai Maksimum Atribut Train:")
print(max_val)

# Ambil sampel acak menggunakan index yang sama untuk membandingkan sebelum vs sesudah
sample_indices = train_data.sample(5, random_state=42).index

print("\n--- Sampel 5 Baris train_data Sebelum Normalisasi (Acak) ---")
print(train_data.loc[sample_indices])

print("\n--- Sampel 5 Baris train_data Setelah Normalisasi (Acak) ---")
print(norm_train_data.loc[sample_indices])
