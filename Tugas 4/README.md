# 🚢 Praktikum Validasi Model (Model Validation) - Titanic Dataset

Repositori ini berisi pengerjaan tugas dan latihan **Pertemuan 5 Mata Kuliah Web Data Mining (Assignment #5 – Validation Model)**, yang berfokus pada tahapan validasi model klasifikasi menggunakan algoritma **k-Nearest Neighbors (k-NN, k=3)** dengan tiga metode pengujian:
1. **Hold-out Method (70% - 30%)**
2. **K-Fold Cross Validation (k=10)**
3. **Leave-One-Out (LOO) Cross Validation**

Dataset yang digunakan adalah **Titanic Dataset** (`titanic.csv`) dengan fitur numerik dan kategorikal (`Sex`, `Age`, `Pclass`, `Fare`) serta target kelas keselamatan (`Survived`).

---

## 📁 Struktur Direktori

```text
Tugas 4/
├── data/
│   ├── titanic.csv                     # Dataset utama Titanic (891 baris)
│   ├── titanic_test.csv                # Dataset pengujian tambahan
│   └── titanic_testlabel.csv           # Label pengujian tambahan
├── jawaban/
│   ├── 01_load_dataset.py              # Soal 1: Load dataset titanic.csv
│   ├── 02_validation_methods.py        # Soal 2: Skema pembagian validasi (Hold-out, K-Fold, LOO)
│   ├── 03_train_data_fitur.py          # Soal 3: Ekstraksi fitur & imputasi missing Age mean per class
│   ├── 04_label_target.py              # Soal 4: Ekstraksi kolom target Survived
│   ├── 05_normalisasi_train_minmax.py  # Soal 5: Normalisasi Min-Max (0-1) data latih & catat min-max
│   ├── 06_normalisasi_test_minmax.py   # Soal 6: Normalisasi data uji dengan parameter train (No Data Leakage)
│   ├── 07_knn_validation.py            # Soal 7: Evaluasi lengkap k-NN (k=3) untuk semua metode validasi
│   ├── 07a_knn_holdout.py              # Soal 7.a: k-NN (k=3) khusus Hold-out Method (70% - 30%)
│   ├── 07b_knn_kfold.py                # Soal 7.b: k-NN (k=3) khusus K-Fold Cross Validation (k=10)
│   └── 07c_knn_loo.py                  # Soal 7.c: k-NN (k=3) khusus Leave-One-Out (LOO)
└── README.md
```

---

## 📋 Ringkasan Pengerjaan Soal (1 Soal 1 File)

| No | File Jawaban | Tugas / Soal | Penjelasan Singkat |
|:--:|:---|:---|:---|
| **1** | [`01_load_dataset.py`](jawaban/01_load_dataset.py) | **1. Load Dataset** | Membaca file `titanic.csv` dan menampilkan ringkasan informasi serta sampel data. |
| **2** | [`02_validation_methods.py`](jawaban/02_validation_methods.py) | **2. Metode Validasi** | Menjelaskan dan menampilkan pembagian data untuk Hold-out (70:30), K-Fold ($k=10$), dan LOO ($N=891$). |
| **3** | [`03_train_data_fitur.py`](jawaban/03_train_data_fitur.py) | **3. Ekstraksi Fitur & Imputasi** | Mengambil fitur `Sex`, `Age`, `Pclass`, `Fare`, melakukan encoding `Sex`, dan imputasi missing value `Age` dengan rata-rata (*mean*) dari masing-masing kelas target `Survived`. |
| **4** | [`04_label_target.py`](jawaban/04_label_target.py) | **4. Label Kelas** | Mengambil kolom target `Survived` (`0` = Tidak Selamat, `1` = Selamat) dan menampilkan distribusi frekuensinya. |
| **5** | [`05_normalisasi_train_minmax.py`](jawaban/05_normalisasi_train_minmax.py) | **5. Normalisasi Min-Max Train** | Melakukan normalisasi skala $[0, 1]$ pada data latih serta mencatat parameter nilai minimum ($x_{\min}$) dan maksimum ($x_{\max}$) setiap fitur. |
| **6** | [`06_normalisasi_test_minmax.py`](jawaban/06_normalisasi_test_minmax.py) | **6. Normalisasi Min-Max Test** | Melakukan penskalaan data uji menggunakan parameter $x_{\min}$ dan $x_{\max}$ dari data latih (Langkah 5) untuk mencegah kebocoran data (*data leakage*). |
| **7** | [`07_knn_validation.py`](jawaban/07_knn_validation.py) | **7. Klasifikasi k-NN (k=3) Lengkap** | Menjalankan seluruh pipeline k-NN ($k=3$) pada ketiga metode validasi (Hold-out, K-Fold, LOO) serta menampilkan tabel perbandingan error ratio dan akurasi. |
| **7.a** | [`07a_knn_holdout.py`](jawaban/07a_knn_holdout.py) | **7.a k-NN Hold-out** | Klasifikasi k-NN ($k=3$) khusus skema Hold-out (70% train : 30% test). |
| **7.b** | [`07b_knn_kfold.py`](jawaban/07b_knn_kfold.py) | **7.b k-NN K-Fold** | Klasifikasi k-NN ($k=3$) khusus skema 10-Fold Cross Validation beserta rincian per fold. |
| **7.c** | [`07c_knn_loo.py`](jawaban/07c_knn_loo.py) | **7.c k-NN Leave-One-Out** | Klasifikasi k-NN ($k=3$) khusus skema Leave-One-Out (891 iterasi). |

---

## 📐 Penjelasan Metode & Rumus yang Digunakan

### 1. Imputasi Missing Value `Age` Berbasis Mean Kelas Target
Nilai kosong (*NaN*) pada kolom umur `Age` diisi menggunakan rata-rata umur penumpang dari kelas keselamatan yang sama pada data latih:
$$\text{Age}_{\text{new}} = \begin{cases} \mu_{\text{Age}\mid\text{Survived}=0}, & \text{jika Survived}=0 \text{ dan Age}=\text{NaN} \\ \mu_{\text{Age}\mid\text{Survived}=1}, & \text{jika Survived}=1 \text{ dan Age}=\text{NaN} \end{cases}$$

### 2. Normalisasi Min-Max (0 - 1) & Pencegahan Data Leakage
Mengubah skala nilai fitur ke rentang $[0, 1]$:
$$x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$
> **Catatan Penting:** Parameter $x_{\min}$ dan $x_{\max}$ dihitung **hanya** dari data latih (*train data*). Data uji (*test data*) kemudian ditransformasikan menggunakan nilai min dan max dari data latih tersebut agar tidak terjadi kebocoran informasi data uji ke dalam model (*Data Leakage*).

### 3. Evaluasi Kinerja Model (k-NN, k=3)
- **Error Ratio:**
  $$\text{Error Ratio} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \mathbb{I}(y_i \neq \hat{y}_i)$$
- **Akurasi:**
  $$\text{Akurasi} = 1 - \text{Error Ratio}$$

---

## 📊 Hasil Evaluasi Model

| Metode Validasi | Jumlah Data Uji | Error Ratio | Akurasi |
|:---|:---:|:---:|:---:|
| **Hold-out Method (70% - 30%)** | 268 baris | **0.1343** (13.43%) | **0.8657** (86.57%) |
| **K-Fold Cross Validation ($k=10$)** | 10 $\times \sim$89 baris | **0.1560** (15.60%) | **0.8440** (84.40%) |
| **Leave-One-Out (LOO)** | 891 baris | **0.1605** (16.05%) | **0.8395** (83.95%) |

---

## 🚀 Cara Menjalankan

Masuk ke folder jawaban:
```powershell
cd "Tugas 4\jawaban"
```

Jalankan script sesuai kebutuhan:
```powershell
# Langkah 1 s.d. 6
python 01_load_dataset.py
python 02_validation_methods.py
python 03_train_data_fitur.py
python 04_label_target.py
python 05_normalisasi_train_minmax.py
python 06_normalisasi_test_minmax.py

# Langkah 7 (Menjalankan semua validasi sekaligus):
python 07_knn_validation.py

# Atau menjalankan validasi tertentu secara individual:
python 07a_knn_holdout.py
python 07b_knn_kfold.py
python 07c_knn_loo.py
```
