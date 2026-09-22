import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, '..', 'data', 'titanic.csv')
if not os.path.exists(csv_path):
    csv_path = 'data/titanic.csv'
if not os.path.exists(csv_path):
    csv_path = 'titanic.csv'

dataset = pd.read_csv(csv_path)

features = ["Sex", "Age", "Pclass", "Fare"]
target = "Survived"

# Ekstraksi fitur yang dibutuhkan serta target kelas
data = dataset[features + [target]].copy()
data["Sex"] = data["Sex"].map({"female": 0, "male": 1})
data["Fare"] = data["Fare"].fillna(data["Fare"].median())

# Pisahkan data latih (70%) dan data uji (30%) dengan stratified sampling agar proporsi kelas seimbang
train_df, test_df = train_test_split(
    data, test_size=0.30, random_state=42, stratify=data[target]
)

train_copy = train_df.copy()
test_copy = test_df.copy()

# Imputasi missing value Age menggunakan nilai mean per kelas dari data latih (mencegah data leakage)
mean_age_train = train_copy.groupby(target)["Age"].mean()
overall_mean_train = train_copy["Age"].mean()

for cls in mean_age_train.index:
    mask_train = (train_copy[target] == cls) & (train_copy["Age"].isna())
    train_copy.loc[mask_train, "Age"] = mean_age_train[cls]
train_copy["Age"] = train_copy["Age"].fillna(overall_mean_train)

for cls in mean_age_train.index:
    mask_test = (test_copy[target] == cls) & (test_copy["Age"].isna())
    test_copy.loc[mask_test, "Age"] = mean_age_train[cls]
test_copy["Age"] = test_copy["Age"].fillna(overall_mean_train)

# Ekstraksi matriks fitur dan vektor target
X_tr = train_copy[features].values.astype(float)
y_tr = train_copy[target].values
X_te = test_copy[features].values.astype(float)
y_te = test_copy[target].values

# Catat nilai min dan max dari data latih untuk normalisasi Min-Max
min_vals = np.min(X_tr, axis=0)
max_vals = np.max(X_tr, axis=0)
range_vals = np.where(max_vals - min_vals == 0, 1.0, max_vals - min_vals)

# Terapkan penskalaan rentang 0-1 pada data latih dan data uji dengan parameter acuan data latih
X_tr_norm = (X_tr - min_vals) / range_vals
X_te_norm = (X_te - min_vals) / range_vals

# Latih model klasifikasi k-NN dengan jumlah tetangga terdekat k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_tr_norm, y_tr)

# Prediksi label kelas pada data uji dan hitung tingkat kesalahan klasifikasi (Error Ratio)
y_pred = knn.predict(X_te_norm)
error_ratio = np.mean(y_pred != y_te)
accuracy = 1.0 - error_ratio

print("=== 7.a Klasifikasi k-NN (k=3) dengan Hold-out Method (70% - 30%) ===")
print(f"Jumlah Data Latih : {len(train_df)} baris")
print(f"Jumlah Data Uji   : {len(test_df)} baris\n")

print("--- Parameter Min & Max Acuan Data Latih ---")
for feat, mi, ma in zip(features, min_vals, max_vals):
    print(f"  {feat:<10}: Min = {mi:.2f}, Max = {ma:.2f}")

print("\n--- Hasil Evaluasi Kinerja Model ---")
print(f"Error Ratio : {error_ratio:.4f} ({error_ratio * 100:.2f}%)")
print(f"Akurasi     : {accuracy:.4f} ({accuracy * 100:.2f}%)")

# Ambil sampel acak menggunakan index yang sama untuk membandingkan prediksi vs label aktual
rng = np.random.RandomState(42)
sample_eval_idx = rng.choice(len(y_te), size=10, replace=False)

eval_sample_df = pd.DataFrame({
    "Label Aktual": y_te[sample_eval_idx],
    "Hasil Prediksi": y_pred[sample_eval_idx],
    "Status": np.where(y_te[sample_eval_idx] == y_pred[sample_eval_idx], "Tepat", "Salah")
})

print("\n--- Sampel 10 Hasil Prediksi Data Uji (Acak) ---")
print(eval_sample_df.to_string(index=False))
