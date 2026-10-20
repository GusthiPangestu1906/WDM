import os
import pandas as pd

def get_csv_path(filename="transaction.csv"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "..", "data", filename),
        os.path.join(script_dir, "data", filename),
        os.path.join(script_dir, filename),
        os.path.join(os.getcwd(), "Tugas 9", "data", filename),
        os.path.join(os.getcwd(), "Tugas 7", "data", filename),
        os.path.join(os.getcwd(), "data", filename),
        os.path.join(os.getcwd(), filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    raise FileNotFoundError(f"File {filename} tidak ditemukan.")

def load_dataset():
    return pd.read_csv(get_csv_path("transaction.csv"))
