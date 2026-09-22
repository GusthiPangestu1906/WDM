import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
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

def evaluate_fold(df_train, df_test, k_neighbors=3):
    train_copy = df_train.copy()
    test_copy = df_test.copy()

    # Hitung rata-rata usia per kelas pada data latih lipatan ini
    mean_age_train = train_copy.groupby(target)["Age"].mean()
    overall_mean_train = train_copy["Age"].mean()

    # Imputasi missing value Age dengan parameter dari lipatan train
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

    # Penskalaan fitur Min-Max dengan parameter acuan data latih lipatan ini
    min_vals = np.min(X_tr, axis=0)
    max_vals = np.max(X_tr, axis=0)
    range_vals = np.where(max_vals - min_vals == 0, 1.0, max_vals - min_vals)

    X_tr_norm = (X_tr - min_vals) / range_vals
    X_te_norm = (X_te - min_vals) / range_vals

    # Latih model k-NN k=3 dan ukur tingkat kesalahan klasifikasi
    knn = KNeighborsClassifier(n_neighbors=k_neighbors)
    knn.fit(X_tr_norm, y_tr)

    y_pred = knn.predict(X_te_norm)
    return np.mean(y_pred != y_te)

# Inisialisasi skema 10-Fold Cross Validation dengan pengacakan terkontrol
kf = KFold(n_splits=10, shuffle=True, random_state=42)
errors_kfold = []

print("=== 7.b Klasifikasi k-NN (k=3) dengan K-Fold Cross Validation (k=10) ===")
print(f"{'Fold':<8} | {'Jumlah Train':<14} | {'Jumlah Test':<13} | {'Error Ratio':<13} | {'Akurasi':<10}")
print("-" * 65)

for fold, (train_idx, test_idx) in enumerate(kf.split(data), 1):
    train_fold = data.iloc[train_idx]
    test_fold = data.iloc[test_idx]

    err = evaluate_fold(train_fold, test_fold, k_neighbors=3)
    errors_kfold.append(err)
    print(f"Fold {fold:02d}   | {len(train_fold):<14} | {len(test_fold):<13} | {err:<13.4f} | {1 - err:<10.4f}")

mean_err = np.mean(errors_kfold)
mean_acc = 1.0 - mean_err

print("-" * 65)
print(f"Rata-rata K-Fold Error Ratio : {mean_err:.4f} ({mean_err * 100:.2f}%)")
print(f"Rata-rata K-Fold Akurasi     : {mean_acc:.4f} ({mean_acc * 100:.2f}%)")
