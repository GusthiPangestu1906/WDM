import os
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, '..', 'data', 'titanic.csv')
if not os.path.exists(csv_path):
    csv_path = 'data/titanic.csv'
if not os.path.exists(csv_path):
    csv_path = 'titanic.csv'

dataset = pd.read_csv(csv_path)

# Ekstraksi label target kelas biner keselamatan
label = dataset["Survived"]

print("=== 4. Ekstraksi Label Kelas Target (Survived) ===")
print(f"Kolom Target: {label.name}")
print(f"Total Data  : {len(label)} baris\n")

print("--- Distribusi Frekuensi Kelas Target ---")
distribusi = label.value_counts().sort_index()
for cls, count in distribusi.items():
    keterangan = "Tidak Selamat (0)" if cls == 0 else "Selamat (1)"
    persentase = (count / len(label)) * 100
    print(f"  Kelas {cls} ({keterangan:<17}): {count:>3} penumpang ({persentase:.2f}%)")

# Ambil sampel acak untuk melihat variasi sebaran nilai label target
sample_indices = label.sample(10, random_state=42).index

print("\n--- Sampel 10 Nilai Label Target (Acak) ---")
print(label.loc[sample_indices].to_string())
