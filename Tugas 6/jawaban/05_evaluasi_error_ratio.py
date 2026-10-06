import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, LeaveOneOut
from data_loader import load_heart_data, SigmoidalScaler, print_df

df, X, y = load_heart_data()

scalers = {
    "Min-Max (0-1)": MinMaxScaler(),
    "Z-Score (Standard)": StandardScaler(),
    "Sigmoidal": SigmoidalScaler()
}

validators = {
    "4-Fold CV": KFold(n_splits=4, shuffle=True, random_state=42),
    "10-Fold CV": KFold(n_splits=10, shuffle=True, random_state=42),
    "Leave-One-Out (LOO)": LeaveOneOut()
}

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "k-NN (k=1)": KNeighborsClassifier(n_neighbors=1),
    "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "k-NN (k=9)": KNeighborsClassifier(n_neighbors=9)
}

print("Sedang memproses seluruh 36 kombinasi eksperimen...")

# Evaluasi seluruh kombinasi eksperimen
records = []
for v_name, val in validators.items():
    for s_name, sc in scalers.items():
        for m_name, mdl in models.items():
            errs = []
            for tr, te in val.split(X):
                s = sc.__class__()
                X_tr, X_te = s.fit_transform(X[tr]), s.transform(X[te])
                clf = mdl.__class__(**mdl.get_params()).fit(X_tr, y[tr])
                errs.append(np.mean(clf.predict(X_te) != y[te]))
            records.append({
                "Validasi": v_name,
                "Normalisasi": s_name,
                "Model": m_name,
                "Error Ratio (%)": round(np.mean(errs) * 100, 2)
            })

df_res = pd.DataFrame(records)

# 1. Matriks Rekapitulasi Error Ratio (%)
pivot = df_res.pivot(index=["Validasi", "Normalisasi"], columns="Model", values="Error Ratio (%)")
pivot_formatted = pivot.map(lambda x: f"{x:.2f}%")
print_df(pivot_formatted, title="Matriks Rekapitulasi Error Ratio (%) Seluruh Percobaan", include_index=True)

# 2. Top 3 Model dengan Error Ratio Terendah
top3 = df_res.sort_values(by="Error Ratio (%)").head(3).copy()
top3["Error Ratio (%)"] = top3["Error Ratio (%)"].map(lambda x: f"{x:.2f}%")
print_df(top3, title="Top 3 Model Terbaik (Error Ratio Terendah)")
