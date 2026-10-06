import pandas as pd
from data_loader import get_csv_path, load_heart_data, COLUMN_NAMES, print_df

# 1. Baca data mentah & data hasil encoding
df_raw = pd.read_csv(get_csv_path(), header=None, names=COLUMN_NAMES, sep=r'\s*,\s*', engine='python')
df_clean, X, y = load_heart_data()

# 2. Ringkasan dataset & sebaran target
info = pd.DataFrame([
    {"Informasi": "Jumlah Baris", "Nilai": f"{len(df_clean)} baris"},
    {"Informasi": "Jumlah Fitur", "Nilai": f"{X.shape[1]} atribut"},
    {"Informasi": "Target 0 (Absence)", "Nilai": f"{sum(y == 0)} sampel ({(sum(y == 0) / len(y)) * 100:.1f}%)"},
    {"Informasi": "Target 1 (Presence)", "Nilai": f"{sum(y == 1)} sampel ({(sum(y == 1) / len(y)) * 100:.1f}%)"},
])
print_df(info, title="Ringkasan Dataset & Distribusi Target")

# 3. Sampel perbandingan nilai sebelum vs sesudah encoding
cols = ["age", "sex", "chest_pain_type", "resting_ecg", "target"]
print_df(df_raw[cols].head(3), title="Sampel Data Sebelum Encoding (Data Mentah)")
print_df(df_clean[cols].head(3), title="Sampel Data Setelah Encoding (Numerik)")
