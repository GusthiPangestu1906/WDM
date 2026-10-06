# 🫀 Praktikum Evaluasi Model Klasifikasi - Heart Disease Dataset

Repositori ini berisi pengerjaan tugas dan latihan **Tugas 6 Mata Kuliah Web Data Mining**, yang berfokus pada evaluasi performa komparatif antara algoritma **k-Nearest Neighbors (k-NN)** dan **Decision Tree Classifier** pada dataset **UCI Heart Disease** menggunakan variasi teknik **Normalisasi Data** dan **Skema Validasi Silang (Cross Validation)**.

---

## 📁 Struktur Direktori

```text
Tugas 6/
├── data/
│   └── heart.csv                   # Dataset UCI Heart Disease (270 sampel, 14 atribut)
├── jawaban/
│   ├── data_loader.py              # Helper pemuat dataset, encoding, custom SigmoidalScaler & print_df
│   ├── 01_load_dan_encoding.py     # Soal 1: Load dataset & konversi data kategorikal ke numerik
│   ├── 02_normalisasi_data.py      # Soal 2: Penerapan 3 teknik normalisasi (Min-Max, Z-Score, Sigmoidal)
│   ├── 03_model_klasifikasi.py     # Soal 3: Setup model klasifikasi (k-NN k=1,5,9 dan Decision Tree)
│   ├── 04_skema_validasi.py        # Soal 4: Konfigurasi skema validasi (4-Fold, 10-Fold, dan LOO)
│   └── 05_evaluasi_error_ratio.py  # Soal 5: Evaluasi komparatif 36 eksperimen & matriks error ratio
└── README.md
```

---

## 📋 Ringkasan Pengerjaan Tiap Soal

| File Jawaban | Topik | Penjelasan Singkat |
| :--- | :--- | :--- |
| [`01_load_dan_encoding.py`](jawaban/01_load_dan_encoding.py) | **Load Dataset & Encoding** | Memuat dataset `heart.csv`, mendeteksi dimensi (270 baris x 14 kolom), melakukan mapping nilai kategorikal ke numerik, dan menghitung persentase target (`Absence`: 55.6%, `Presence`: 44.4%). |
| [`02_normalisasi_data.py`](jawaban/02_normalisasi_data.py) | **Normalisasi Fitur** | Menerapkan 3 metode penskalaan fitur: **Min-Max (0-1)**, **Z-Score (Standarisasi)**, dan **Sigmoidal**, serta menampilkan perbandingan nilai sebelum dan sesudah transformasi. |
| [`03_model_klasifikasi.py`](jawaban/03_model_klasifikasi.py) | **Setup Model Klasifikasi** | Mengonfigurasi 4 model: **k-NN ($k=1$)**, **k-NN ($k=5$)**, **k-NN ($k=9$)**, dan **Decision Tree Classifier**, serta melakukan pengujian dasar split 70:30. |
| [`04_skema_validasi.py`](jawaban/04_skema_validasi.py) | **Skema Validasi Model** | Mengonfigurasi 3 teknik validasi silang: **4-Fold Cross Validation**, **10-Fold Cross Validation**, dan **Leave-One-Out (LOO)**. |
| [`05_evaluasi_error_ratio.py`](jawaban/05_evaluasi_error_ratio.py) | **Evaluasi Error Ratio** | Menjalankan seluruh **36 kombinasi eksperimen** (3 Validasi x 3 Normalisasi x 4 Model) tanpa data leakage, dan menyajikan matriks rekapitulasi error ratio. |

---

## 📊 Matriks Rekapitulasi Error Ratio (%)

Hasil perhitungan rata-rata *Error Ratio* (%) untuk seluruh 36 kombinasi eksperimen:

| Validasi | Normalisasi | Decision Tree | k-NN ($k=1$) | k-NN ($k=5$) | k-NN ($k=9$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **4-Fold CV** | Min-Max (0-1) | 26.69% | 23.68% | 19.98% | 20.36% |
| | Z-Score (Standard) | 26.69% | 20.75% | 15.92% | **15.18%** 🏆 |
| | Sigmoidal | 27.06% | 21.12% | 17.41% | 16.29% |
| **10-Fold CV** | Min-Max (0-1) | 24.44% | 22.96% | 19.26% | 19.63% |
| | Z-Score (Standard) | 24.81% | 22.59% | 17.41% | **16.67%** |
| | Sigmoidal | 24.81% | 22.96% | 17.78% | 17.04% |
| **Leave-One-Out (LOO)** | Min-Max (0-1) | 27.41% | 24.07% | 20.00% | 20.00% |
| | Z-Score (Standard) | 27.41% | 23.33% | 18.52% | **17.04%** |
| | Sigmoidal | 27.78% | 24.07% | 18.52% | 17.78% |

---

## 🏆 Temuan & Analisis

1. **Model Terbaik**: Kombinasi **k-NN ($k=9$)** dengan normalisasi **Z-Score (Standard)** pada validasi **4-Fold CV** memberikan performa terbaik dengan nilai *Error Ratio* terendah sebesar **15.18%** (Akurasi **84.82%**).
2. **Pengaruh Normalisasi**: Algoritma k-NN sangat sensitif terhadap skala fitur. Normalisasi **Z-Score** dan **Sigmoidal** secara konsisten menghasilkan error yang jauh lebih rendah dibandingkan data tanpa normalisasi maupun Min-Max.
3. **Pengaruh Nilai $k$**: Peningkatan nilai tetangga pada k-NN ($k=5$ dan $k=9$) mampu menghaluskan batas keputusan (*decision boundary*) dan mereduksi dampak *noise* jika dibandingkan dengan $k=1$.
4. **Decision Tree**: Performa Decision Tree relatif independen terhadap teknik normalisasi (karena pemisahan nilai fitur berdasarkan *threshold* bertingkat), dengan rata-rata error berkisar antara 24% hingga 27%.
