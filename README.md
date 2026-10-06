# 🌐 Web Data Mining (WDM)

Repositori ini berisi kumpulan tugas, praktikum, dan materi mata kuliah **Web Data Mining** dengan studi kasus analisis, *preprocessing* data, dan pemodelan *machine learning* menggunakan **Python**, **Pandas**, **NumPy**, dan **Scikit-Learn**.

---

## 📁 Struktur Direktori Repositori

```text
WDM/
├── Tugas 1/                        # Praktikum Pertemuan 2: Data Preprocessing & Visualisasi
│   ├── data/
│   │   └── titanic.csv             # Dataset Titanic
│   ├── jawaban/                    # Script jawaban soal nomor 1 - 10
│   ├── materi/                     # Slide materi Pertemuan 2
│   ├── output/                     # Hasil visualisasi grafik
│   └── README.md                   # Dokumentasi lengkap pengerjaan Tugas 1
│
├── Tugas 2/                        # Praktikum Pertemuan 3: Imputasi Missing Value & Normalisasi Data
│   ├── data/
│   │   └── titanic.csv             # Dataset Titanic
│   ├── jawaban/                    # Script jawaban soal nomor 1 - 8
│   ├── materi/                     # Slide materi Pertemuan 3
│   └── README.md                   # Dokumentasi lengkap pengerjaan Tugas 2
│
├── Tugas 3/                        # Praktikum Pertemuan 4: Klasifikasi k-NN dengan Pembobotan Jarak
│   ├── data/                       # Dataset train, test, dan testlabel
│   ├── jawaban/                    # Script jawaban nomor 1 - 10
│   └── README.md                   # Dokumentasi lengkap pengerjaan Tugas 3
│
├── Tugas 4/                        # Praktikum Pertemuan 5: Validasi Model (Hold-out, K-Fold, LOO) & k-NN
│   ├── data/                       # Dataset Titanic
│   ├── jawaban/                    # Script jawaban soal 1 - 7 (dan varian validasi)
│   └── README.md                   # Dokumentasi lengkap pengerjaan Tugas 4
│
├── Tugas 5/                        # Praktikum Pertemuan 6: Decision Tree Classifier & Visualisasi Node
│   ├── data/                       # Dataset Titanic
│   ├── jawaban/                    # Script decision_tree_classifier.py & visualisasi pohon
│   └── README.md                   # Dokumentasi lengkap pengerjaan Tugas 5
│
├── Tugas 6/                        # Praktikum Pertemuan 7: Evaluasi Klasifikasi (Normalisasi & Validasi Silang)
│   ├── data/                       # Dataset UCI Heart Disease
│   ├── jawaban/                    # Script jawaban soal 1 - 5 & data_loader
│   └── README.md                   # Dokumentasi lengkap pengerjaan Tugas 6
│
├── .gitignore
└── README.md
```

---

## 📚 Daftar Praktikum & Tugas

### 1. [Tugas 1: Data Preprocessing & Visualisasi](Tugas%201/README.md)
Fokus pada pengenalan dataset Titanic, pemahaman dimensi data, ekstraksi fitur (`Name`, `Sex`, `Age`, `Pclass`, `Fare`), pembuatan fitur baru (*Relatives*), analisis frekuensi dan tabulasi silang (*crosstab*), serta visualisasi distribusi data keselamatan menggunakan **Matplotlib** dan **Seaborn**.

### 2. [Tugas 2: Imputasi Missing Value & Normalisasi Data](Tugas%202/README.md)
Fokus pada penanganan nilai kosong (*missing value*) pada fitur numerik (`Age`) menggunakan nilai rata-rata per kelompok kelas (`Survived`), serta penerapan tiga teknik normalisasi fitur numerik:
- **Min-Max (0 - 1)**
- **Z-Score (Standarisasi)**
- **Sigmoida**

### 3. [Tugas 3: Klasifikasi k-NN dengan Pembobotan Jarak](Tugas%203/README.md)
Fokus pada pemodelan klasifikasi menggunakan algoritma **k-Nearest Neighbors (k-NN)** dengan pembobotan jarak (*weighted distance*) pada data uji terpisah serta evaluasi *Precision Ratio* dan *Error Ratio* untuk variasi nilai $k = 1$ hingga $15$.

### 4. [Tugas 4: Validasi Model (Model Validation) & k-NN](Tugas%204/README.md)
Fokus pada pengujian generalisasi model k-NN ($k=3$) menggunakan tiga skema validasi:
- **Hold-out Method (70% - 30%)**
- **K-Fold Cross Validation (k = 10)**
- **Leave-One-Out (LOO) Cross Validation**

### 5. [Tugas 5: Klasifikasi Decision Tree & Visualisasi Node](Tugas%205/README.md)
Fokus pada klasifikasi menggunakan algoritma **Decision Tree Classifier**, penanganan *overfitting* melalui pembatasan kedalaman pohon (*pruning* dengan `max_depth=3`), analisis kepentingan fitur (*Feature Importance*), serta visualisasi diagram node pohon resolusi tinggi.

### 6. [Tugas 6: Evaluasi Model Klasifikasi (Normalisasi & Validasi Silang)](Tugas%206/README.md)
Fokus pada evaluasi komparatif performa algoritma **k-NN** ($k=1, 5, 9$) dan **Decision Tree Classifier** pada dataset **UCI Heart Disease**, menguji kombinasi 3 teknik normalisasi data (**Min-Max**, **Z-Score**, **Sigmoidal**) serta 3 metode validasi (**4-Fold CV**, **10-Fold CV**, **Leave-One-Out / LOO**) dengan total 36 kombinasi eksperimen tanpa *data leakage*.

---

## 🚀 Persyaratan & Instalasi

Pastikan telah menginstall library Python yang dibutuhkan:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```
