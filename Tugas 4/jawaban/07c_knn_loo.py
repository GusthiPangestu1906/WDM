import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
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

# Ekstraksi fitur dan target
data = dataset[features + [target]].copy()
data["Sex"] = data["Sex"].map({"female": 0, "male": 1})
data["Fare"] = data["Fare"].fillna(data["Fare"].median())

# Inisialisasi Leave-One-Out di mana setiap iterasi menguji tepat 1 sampel
loo = LeaveOneOut()
total_samples = len(data)
errors_loo = []

print("=== 7.c Klasifikasi k-NN (k=3) dengan Leave-One-Out (LOO) ===")
print(f"Total Sampel Data (N) : {total_samples}")
print("Memproses LOO (menjalankan 891 pengujian satu per satu)...")

start_time = time.time()

for idx, (train_idx, test_idx) in enumerate(loo.split(data), 1):
    train_copy = data.iloc[train_idx].copy()
    test_copy = data.iloc[test_idx].copy()

    # Hitung rata-rata usia per kelas pada data latih (N - 1 sampel)
    mean_age_train = train_copy.groupby(target)["Age"].mean()
    overall_mean_train = train_copy["Age"].mean()

    # Imputasi missing value Age dengan parameter dari data latih
    for cls in mean_age_train.index:
        mask_tr = (train_copy[target] == cls) & (train_copy["Age"].isna())
        train_copy.loc[mask_tr, "Age"] = mean_age_train[cls]
    train_copy["Age"] = train_copy["Age"].fillna(overall_mean_train)

    for cls in mean_age_train.index:
        mask_te = (test_copy[target] == cls) & (test_copy["Age"].isna())
        test_copy.loc[mask_te, "Age"] = mean_age_train[cls]
    test_copy["Age"] = test_copy["Age"].fillna(overall_mean_train)

    X_tr = train_copy[features].values.astype(float)
    y_tr = train_copy[target].values
    X_te = test_copy[features].values.astype(float)
    y_te = test_copy[target].values

    # Penskalaan fitur Min-Max dengan acuan data latih
    min_vals = np.min(X_tr, axis=0)
    max_vals = np.max(X_tr, axis=0)
    range_vals = np.where(max_vals - min_vals == 0, 1.0, max_vals - min_vals)

    X_tr_norm = (X_tr - min_vals) / range_vals
    X_te_norm = (X_te - min_vals) / range_vals

    # Latih model k-NN k=3 dan catat ketepatan prediksi pada 1 sampel uji
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_tr_norm, y_tr)

    y_pred = knn.predict(X_te_norm)
    errors_loo.append(int(y_pred[0] != y_te[0]))

    # Tampilkan pembaruan progres berkala ke terminal
    if idx % 150 == 0 or idx == total_samples:
        print(f"  Progres: {idx}/{total_samples} sampel selesai dievaluasi...")

duration = time.time() - start_time
mean_err_loo = np.mean(errors_loo)
accuracy_loo = 1.0 - mean_err_loo

print("\n--- Hasil Evaluasi Leave-One-Out (LOO) ---")
print(f"Waktu Eksekusi            : {duration:.2f} detik")
print(f"Total Prediksi Salah      : {np.sum(errors_loo)} dari {total_samples}")
print(f"Leave-One-Out Error Ratio : {mean_err_loo:.4f} ({mean_err_loo * 100:.2f}%)")
print(f"Leave-One-Out Akurasi     : {accuracy_loo:.4f} ({accuracy_loo * 100:.2f}%)")
