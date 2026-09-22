import os
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, '..', 'data', 'titanic.csv')
if not os.path.exists(csv_path):
    csv_path = 'data/titanic.csv'
if not os.path.exists(csv_path):
    csv_path = 'titanic.csv'

dataset = pd.read_csv(csv_path)

features = ["Sex", "Age", "Pclass", "Fare"]
target = "Survived"

# Ekstraksi kolom fitur numerik dan target
data = dataset[features + [target]].copy()

# Encoding fitur biner kategorikal 'Sex' (female: 0, male: 1)
data["Sex"] = data["Sex"].map({"female": 0, "male": 1})

# Mengatasi missing value pada 'Fare' menggunakan nilai median
data["Fare"] = data["Fare"].fillna(data["Fare"].median())

# Salin data sebelum imputasi untuk bahan perbandingan evaluasi
data_before = data.copy()

# Hitung rata-rata usia per kelas keselamatan (Survived)
mean_age_per_class = data.groupby(target)["Age"].mean()
overall_mean_age = data["Age"].mean()

print("=== 3. Ekstraksi Fitur & Imputasi Missing Value 'Age' ===")
print(f"Fitur Terpilih: {features}")
print(f"Missing Value 'Age' Sebelum Imputasi : {data['Age'].isna().sum()} baris")
print(f"Rata-rata Usia Kelas Tidak Selamat (0): {mean_age_per_class[0]:.2f} tahun")
print(f"Rata-rata Usia Kelas Selamat (1)      : {mean_age_per_class[1]:.2f} tahun")

# Isi missing value kolom Age dengan mean kelas masing-masing
for cls in mean_age_per_class.index:
    mask = (data[target] == cls) & (data["Age"].isna())
    data.loc[mask, "Age"] = mean_age_per_class[cls]

# Nilai rata-rata keseluruhan sebagai fallback jika ada kelas tak terdefinisi
data["Age"] = data["Age"].fillna(overall_mean_age)

print(f"Missing Value 'Age' Setelah Imputasi  : {data['Age'].isna().sum()} baris")

# Ambil sampel baris yang awalnya bernilai NaN pada kolom Age untuk membandingkan sebelum vs sesudah
sample_missing_indices = data_before[data_before["Age"].isna()].sample(5, random_state=42).index

print("\n--- Sampel 5 Baris Data Sebelum Imputasi (Kondisi Nilai NaN) ---")
print(data_before.loc[sample_missing_indices])

print("\n--- Sampel 5 Baris Data Setelah Imputasi (Berdasarkan Nilai Mean Kelas) ---")
print(data.loc[sample_missing_indices])
