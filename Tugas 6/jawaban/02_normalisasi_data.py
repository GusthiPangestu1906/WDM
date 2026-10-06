import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from data_loader import load_heart_data, SigmoidalScaler, print_df

df, X, y = load_heart_data()
cols = ["age", "resting_blood_pressure", "cholesterol", "max_heart_rate"]
sample = df[cols].head(3)

scalers = {
    "Min-Max (0-1)": MinMaxScaler(),
    "Z-Score (Standard)": StandardScaler(),
    "Sigmoidal": SigmoidalScaler()
}

# Tampilkan nilai fitur sebelum dan sesudah masing-masing normalisasi
print_df(sample.round(1), title="Nilai Fitur Sebelum Normalisasi (Sampel 3 Baris)")
for name, s in scalers.items():
    transformed = pd.DataFrame(s.fit_transform(df[cols]).round(4), columns=cols).head(3)
    print_df(transformed, title=f"Hasil Normalisasi: {name}")
