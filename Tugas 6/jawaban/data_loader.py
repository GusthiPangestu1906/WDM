import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Kelas normalisasi Sigmoidal berbasis nilai z-score fitur
class SigmoidalScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        arr = np.asarray(X)
        self.mean_ = np.mean(arr, axis=0)
        self.std_ = np.std(arr, axis=0)
        self.std_[self.std_ == 0.0] = 1.0  # Cegah pembagian dengan nol
        return self

    def transform(self, X):
        arr = np.asarray(X)
        z = (arr - self.mean_) / self.std_
        return 1.0 / (1.0 + np.exp(-z))

# Atribut kolom standar dataset UCI Heart Disease
COLUMN_NAMES = [
    "age", "sex", "chest_pain_type", "resting_blood_pressure", "cholesterol",
    "fasting_blood_sugar", "resting_ecg", "max_heart_rate",
    "exercise_induced_angina", "oldpeak", "st_slope",
    "major_vessels", "thal", "target"
]

# Aturan mapping konversi nilai kategorikal string ke representasi numerik
ENCODING_RULES = {
    "sex": {"female": 0, "male": 1, "0": 0, "1": 1},
    "chest_pain_type": {
        "typical_angina": 1, "atypical_angina": 2, 
        "non_anginal_pain": 3, "asymptomatic": 4
    },
    "resting_ecg": {
        "normal": 0, "st_t_wave_abnormality": 1, "wave_abnormality": 1, "ventricular_hypertrophy": 2
    },
    "exercise_induced_angina": {"no": 0, "yes": 1, "0": 0, "1": 1},
    "st_slope": {"upsloping": 1, "flat": 2, "downsloping": 3},
    "thal": {"normal": 3, "fixed_defect": 6, "reversable_defect": 7},
    "target": {"Absence": 0, "Presence": 1, "1": 0, "2": 1, 0: 0, 1: 1}
}

# Deteksi lokasi path file heart.csv
def get_csv_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "..", "data", "heart.csv"),
        os.path.join(script_dir, "data", "heart.csv"),
        os.path.join(script_dir, "heart.csv"),
        os.path.join(os.getcwd(), "data", "heart.csv"),
        os.path.join(os.getcwd(), "heart.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    raise FileNotFoundError("File heart.csv tidak ditemukan di direktori data atau jawaban.")

# Fungsi pembacaan dataset heart.csv dan konversi otomatis ke numerik
def load_heart_data():
    csv_file = get_csv_path()
    df = pd.read_csv(csv_file, header=None, names=COLUMN_NAMES, sep=r'\s*,\s*', engine='python')
    
    for col, mapping in ENCODING_RULES.items():
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().map(mapping).fillna(df[col])
            df[col] = pd.to_numeric(df[col])
            
    X = df.drop(columns=["target"]).values.astype(float)
    y = df["target"].values.astype(int)
    return df, X, y

# Fungsi pembantu untuk memformat dan mencetak tabel ASCII bergaris rapi
def print_table(headers, rows, title=None):
    if title:
        print(f"\n[ {title} ]")
    
    s_rows = []
    for r in rows:
        if r == ["---"] or r == "---":
            s_rows.append("---")
        else:
            s_rows.append([str(c) for c in r])

    col_widths = [len(str(h)) for h in headers]
    for row in s_rows:
        if row != "---":
            for idx, cell in enumerate(row):
                if idx < len(col_widths):
                    col_widths[idx] = max(col_widths[idx], len(cell))
                else:
                    col_widths.append(len(cell))
    
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    
    def format_row(row_items):
        cells = []
        for idx, item in enumerate(row_items):
            w = col_widths[idx]
            clean = item.strip().rstrip('%')
            try:
                float(clean)
                cells.append(f" {item:>{w}} ")
            except ValueError:
                cells.append(f" {item:<{w}} ")
        return "|" + "|".join(cells) + "|"

    print(sep)
    print(format_row(headers))
    print(sep)
    for row in s_rows:
        if row == "---":
            print(sep)
        else:
            print(format_row(row))
    print(sep)

# Fungsi pembantu untuk mencetak pandas DataFrame secara otomatis ke format tabel rapi
def print_df(df, title=None, include_index=False):
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if include_index or isinstance(df.index, pd.MultiIndex) or (df.index.name is not None and df.index.name != ""):
        df = df.reset_index()
    headers = [str(c) for c in df.columns]
    rows = [[str(cell) for cell in row] for row in df.values]
    print_table(headers, rows, title=title)
