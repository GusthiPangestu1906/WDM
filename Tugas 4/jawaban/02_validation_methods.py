import os
import pandas as pd
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split

base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, '..', 'data', 'titanic.csv')
if not os.path.exists(csv_path):
    csv_path = 'data/titanic.csv'

dataset = pd.read_csv(csv_path)
target = "Survived"

print("=== 2. Skema Pembagian Metode Validasi Model ===")
print(f"Total Dataset: {len(dataset)} baris\n")

# 2.a Hold-out Method (70% Train, 30% Test)
print("--- 2.a Hold-out Method (70% Train, 30% Test) ---")
train_df, test_df = train_test_split(
    dataset, test_size=0.30, random_state=42, stratify=dataset[target]
)
print(f"Jumlah Data Train (70%) : {len(train_df)} baris")
print(f"Jumlah Data Test  (30%) : {len(test_df)} baris")
print(f"Distribusi Target Train : {dict(train_df[target].value_counts())}")
print(f"Distribusi Target Test  : {dict(test_df[target].value_counts())}\n")

# 2.b K-Fold Cross Validation (k = 10)
print("--- 2.b K-Fold Cross Validation (k = 10) ---")
kf = KFold(n_splits=10, shuffle=True, random_state=42)
print(f"Jumlah Lipatan (Folds)  : {kf.get_n_splits(dataset)}")
for fold, (train_idx, test_idx) in enumerate(kf.split(dataset), 1):
    print(f"  Fold {fold:02d} -> Train: {len(train_idx)} baris, Test: {len(test_idx)} baris")
print()

# 2.c Leave-One-Out (LOO) Cross Validation
print("--- 2.c Leave-One-Out (LOO) Cross Validation ---")
loo = LeaveOneOut()
n_splits = loo.get_n_splits(dataset)
print(f"Total Eksperimen (N)    : {n_splits}")
print("Setiap iterasi          : Data Train = N - 1 (890 baris), Data Test = 1 baris")
