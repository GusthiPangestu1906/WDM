import pandas as pd
from sklearn.model_selection import KFold, LeaveOneOut
from data_loader import load_heart_data, print_df

df, X, y = load_heart_data()
n = len(X)

validators = {
    "4-Fold CV": KFold(n_splits=4, shuffle=True, random_state=42),
    "10-Fold CV": KFold(n_splits=10, shuffle=True, random_state=42),
    "Leave-One-Out (LOO)": LeaveOneOut()
}

# Tampilkan ringkasan pembagian data tiap skema validasi
summary = []
for name, val in validators.items():
    tr, te = next(val.split(X))
    summary.append({
        "Skema Validasi": name,
        "Total Folds": val.get_n_splits(X),
        "Porsi Train / Fold": f"{len(tr)} baris ({len(tr)/n*100:.1f}%)",
        "Porsi Test / Fold": f"{len(te)} baris ({len(te)/n*100:.1f}%)"
    })

print_df(pd.DataFrame(summary), title="Konfigurasi Skema Validasi Model")
