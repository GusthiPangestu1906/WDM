import os
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', 'data')

test_path = os.path.join(data_dir, 'titanic_test.csv')
if not os.path.exists(test_path):
    test_path = 'data/titanic_test.csv'

testlabel_path = os.path.join(data_dir, 'titanic_testlabel.csv')
if not os.path.exists(testlabel_path):
    testlabel_path = 'data/titanic_testlabel.csv'

# Identifikasi baris missing value pada test data (Age, Fare)
test_dataset = pd.read_csv(test_path)
test_features = test_dataset[["Age", "Fare"]]
pos_missing_test = test_features.isna().any(axis=1)

# Baca label test
raw_test_label = pd.read_csv(testlabel_path)
target_col = "Survived" if "Survived" in raw_test_label.columns else raw_test_label.columns[0]
test_label = raw_test_label.loc[~pos_missing_test, target_col]

print("=== 6. Label Test dari titanic_testlabel.csv ===")
print(f"File Label Test          : {testlabel_path}")
print(f"Kolom Target Digunakan   : '{target_col}'")
print(f"Jumlah baris raw label   : {len(raw_test_label)}")
print(f"Jumlah test_label valid  : {len(test_label)}")
print(f"\nDistribusi Kelas Test:\n{test_label.value_counts()}")

# Tampilkan sampel data secara acak (random sample)
print("\n--- Sampel 10 Label Secara Acak (Random Sample) ---")
print(test_label.sample(10, random_state=42))
