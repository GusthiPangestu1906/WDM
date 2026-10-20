import os
import pandas as pd

def get_csv_path(filename="titanic.csv"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "..", "data", filename),
        os.path.join(script_dir, "data", filename),
        os.path.join(script_dir, filename),
        os.path.join(os.getcwd(), "data", filename),
        os.path.join(os.getcwd(), "UTS", "data", filename),
        os.path.join(os.getcwd(), filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    raise FileNotFoundError(f"File {filename} tidak ditemukan.")

def load_dataset():
    return pd.read_csv(get_csv_path("titanic.csv"))

def load_test_dataset():
    return pd.read_csv(get_csv_path("titanic_test.csv"))

def load_test_label():
    return pd.read_csv(get_csv_path("titanic_testlabel.csv"))

def load_features():
    df = load_dataset()
    data = df[["Sex", "Age", "Pclass", "Fare", "Survived"]].copy()
    data["Sex"] = data["Sex"].map({"female": 0, "male": 1})
    return data
