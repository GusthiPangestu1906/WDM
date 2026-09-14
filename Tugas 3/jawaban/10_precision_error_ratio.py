import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', 'data')

train_path = os.path.join(data_dir, 'titanic.csv')
test_path = os.path.join(data_dir, 'titanic_test.csv')
testlabel_path = os.path.join(data_dir, 'titanic_testlabel.csv')

if not os.path.exists(train_path):
    train_path = 'data/titanic.csv'
if not os.path.exists(test_path):
    test_path = 'data/titanic_test.csv'
if not os.path.exists(testlabel_path):
    testlabel_path = 'data/titanic_testlabel.csv'

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

# Ambil label pengujian
raw_test_label = pd.read_csv(testlabel_path)
target_col = "Survived" if "Survived" in raw_test_label.columns else raw_test_label.columns[0]
test_label = raw_test_label.loc[~pos_missing_test, target_col]

print("=== 10. Perhitungan Precision Ratio & Error Ratio (k = 1 s.d. 15) ===")
print(f"{'k':<5} | {'Precision Ratio':<17} | {'Error Ratio':<15}")
print("-" * 43)

for k in range(1, 16):
    # Inisialisasi model k-NN dengan weights='distance'
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")

    # Fit data pelatihan
    knn.fit(train_data, train_label)

    # Prediksi hasil kelas
    class_result = knn.predict(test_data)

    # Menghitung precision ratio (akurasi/skor) dan error ratio
    precision_ratio = knn.score(test_data, test_label)
    error_ratio = 1.0 - precision_ratio

    print(f"{k:<5} | {precision_ratio:<17.4f} | {error_ratio:<15.4f}")
