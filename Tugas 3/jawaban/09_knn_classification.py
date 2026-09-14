import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', 'data')

train_path = os.path.join(data_dir, 'titanic.csv')
test_path = os.path.join(data_dir, 'titanic_test.csv')

if not os.path.exists(train_path):
    train_path = 'data/titanic.csv'
if not os.path.exists(test_path):
    test_path = 'data/titanic_test.csv'

# Preprocessing Train Data
dataset = pd.read_csv(train_path)
train_features = dataset[["Age", "Fare"]]
pos_missing_train = train_features.isna().any(axis=1)
train_data = train_features[~pos_missing_train].copy()
train_label = dataset.loc[~pos_missing_train, "Survived"]

min_val = train_data.min()
max_val = train_data.max()
train_data = (train_data - min_val) / (max_val - min_val)

# Preprocessing Test Data
test_dataset = pd.read_csv(test_path)
test_features = test_dataset[["Age", "Fare"]]
pos_missing_test = test_features.isna().any(axis=1)
test_data = test_features[~pos_missing_test].copy()
test_data = (test_data - min_val) / (max_val - min_val)

print("=== 9. Klasifikasi k-NN (k = 1 s.d. 15, weights='distance') ===")
print(f"Data Pelatihan (Train) : {len(train_data)} sampel")
print(f"Data Pengujian (Test)  : {len(test_data)} sampel\n")

print(f"{'k':<5} | {'Prediksi 0 (Tidak Selamat)':<27} | {'Prediksi 1 (Selamat)':<20}")
print("-" * 57)

for k in range(1, 16):
    # Inisialisasi model k-NN dengan bobot distance
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")

    # Fit data pelatihan
    knn.fit(train_data, train_label)

    # Prediksi hasil kelas
    class_result = knn.predict(test_data)

    count_0 = (class_result == 0).sum()
    count_1 = (class_result == 1).sum()
    print(f"{k:<5} | {count_0:<27} | {count_1:<20}")
