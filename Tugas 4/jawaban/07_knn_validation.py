import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split
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


# Pipeline evaluasi mandiri per-split untuk mencegah kebocoran data (Data Leakage)
def evaluate_pipeline(df_train, df_test, k_neighbors=3):
    train_copy = df_train.copy()
    test_copy = df_test.copy()

    # Hitung rata-rata usia per kelas target hanya dari data latih
    mean_age_train = train_copy.groupby(target)["Age"].mean()
    overall_mean_train = train_copy["Age"].mean()

    # Terapkan imputasi pada data latih dan data uji dengan nilai mean data latih
    for cls in mean_age_train.index:
        mask_tr = (train_copy[target] == cls) & (train_copy["Age"].isna())
        train_copy.loc[mask_tr, "Age"] = mean_age_train[cls]
    train_copy["Age"] = train_copy["Age"].fillna(overall_mean_train)

    for cls in mean_age_train.index:
        mask_te = (test_copy[target] == cls) & (test_copy["Age"].isna())
        test_copy.loc[mask_te, "Age"] = mean_age_train[cls]
    test_copy["Age"] = test_copy["Age"].fillna(overall_mean_train)

    # Ekstraksi array fitur dan label target
    X_tr = train_copy[features].values.astype(float)
    y_tr = train_copy[target].values
    X_te = test_copy[features].values.astype(float)
    y_te = test_copy[target].values

    # Catat batas minimum dan maksimum dari data latih
    min_vals = np.min(X_tr, axis=0)
    max_vals = np.max(X_tr, axis=0)
    range_vals = np.where(max_vals - min_vals == 0, 1.0, max_vals - min_vals)

    # Normalisasi Min-Max [0, 1]
    X_tr_norm = (X_tr - min_vals) / range_vals
    X_te_norm = (X_te - min_vals) / range_vals

    # Latih model k-NN k=3 dan ukur tingkat kesalahan prediksi
    knn = KNeighborsClassifier(n_neighbors=k_neighbors)
    knn.fit(X_tr_norm, y_tr)

    y_pred = knn.predict(X_te_norm)
    error_ratio = np.mean(y_pred != y_te)

    return error_ratio, min_vals, max_vals


print("=" * 65)
print("=== 7. KLASIFIKASI k-NN (k=3) PADA SELURUH METODE VALIDASI ===")
print("=" * 65)

# --- 1. Hold-out Method (70% Train, 30% Test) ---
print("\n=== 1. HOLDOUT METHOD (70% - 30%) ===")
train_df, test_df = train_test_split(
    data, test_size=0.30, random_state=42, stratify=data[target]
)

err_holdout, min_holdout, max_holdout = evaluate_pipeline(train_df, test_df, k_neighbors=3)

print("Parameter Min Train (Sex, Age, Pclass, Fare):", np.round(min_holdout, 2))
print("Parameter Max Train (Sex, Age, Pclass, Fare):", np.round(max_holdout, 2))
print(f"Error Ratio : {err_holdout:.4f} ({err_holdout * 100:.2f}%)")
print(f"Akurasi     : {1 - err_holdout:.4f} ({(1 - err_holdout) * 100:.2f}%)")

# --- 2. K-Fold Cross Validation (k=10) ---
print("\n=== 2. K-FOLD CROSS VALIDATION (k=10) ===")
kf = KFold(n_splits=10, shuffle=True, random_state=42)
errors_kfold = []

for fold, (train_idx, test_idx) in enumerate(kf.split(data), 1):
    train_fold = data.iloc[train_idx]
    test_fold = data.iloc[test_idx]

    err, _, _ = evaluate_pipeline(train_fold, test_fold, k_neighbors=3)
    errors_kfold.append(err)
    print(f"  Fold {fold:02d} -> Error Ratio: {err:.4f} | Akurasi: {1 - err:.4f}")

mean_err_kfold = np.mean(errors_kfold)
print(f"\nRata-rata K-Fold Error Ratio : {mean_err_kfold:.4f} ({mean_err_kfold * 100:.2f}%)")
print(f"Rata-rata Akurasi            : {1 - mean_err_kfold:.4f} ({(1 - mean_err_kfold) * 100:.2f}%)")

# --- 3. Leave-One-Out (LOO) Cross Validation ---
print("\n=== 3. LEAVE-ONE-OUT (LOO) CROSS VALIDATION ===")
print("Memproses LOO (menjalankan 891 eksperimen)... Mohon tunggu...")
loo = LeaveOneOut()
errors_loo = []

for train_idx, test_idx in loo.split(data):
    train_loo = data.iloc[train_idx]
    test_loo = data.iloc[test_idx]

    err, _, _ = evaluate_pipeline(train_loo, test_loo, k_neighbors=3)
    errors_loo.append(err)

mean_err_loo = np.mean(errors_loo)
print(f"Total Eksperimen (N)         : {len(errors_loo)}")
print(f"Leave-One-Out Error Ratio    : {mean_err_loo:.4f} ({mean_err_loo * 100:.2f}%)")
print(f"Akurasi LOO                  : {1 - mean_err_loo:.4f} ({(1 - mean_err_loo) * 100:.2f}%)")

# --- Ringkasan Perbandingan Kinerja Seluruh Metode Validasi ---
print("\n" + "=" * 65)
print(f"{'Metode Validasi':<30} | {'Error Ratio':<15} | {'Akurasi':<12}")
print("-" * 65)
print(f"{'Hold-out Method (70% - 30%)':<30} | {err_holdout:<15.4f} | {1 - err_holdout:<12.4f}")
print(f"{'K-Fold Cross Validation (k=10)':<30} | {mean_err_kfold:<15.4f} | {1 - mean_err_kfold:<12.4f}")
print(f"{'Leave-One-Out (LOO)':<30} | {mean_err_loo:<15.4f} | {1 - mean_err_loo:<12.4f}")
print("=" * 65)
