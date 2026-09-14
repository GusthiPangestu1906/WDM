# 🚢 Praktikum Klasifikasi k-Nearest Neighbors (k-NN) - Titanic Dataset

Repositori ini berisi pengerjaan tugas dan latihan **Pertemuan 4 Mata Kuliah Web Data Mining**, yang berfokus pada tahapan pemodelan klasifikasi menggunakan algoritma **k-Nearest Neighbors (k-NN)** dengan pembobotan jarak (*weighted distance*) pada dataset penumpang kapal Titanic.

---

## 📁 Struktur Direktori

```text
Tugas 3/
├── data/
│   ├── titanic.csv                  # Dataset pelatihan (Train dataset)
│   ├── titanic_test.csv             # Dataset pengujian fitur (Test dataset)
│   └── titanic_testlabel.csv        # Label kelas data uji (Ground truth test)
├── jawaban/
│   ├── 01_load_dataset.py           # 1. Membaca dataset pelatihan (train)
│   ├── 02_load_test_dataset.py      # 2. Membaca dataset pengujian (test)
│   ├── 03_train_data_fitur.py       # 3. Ekstraksi fitur train (Age, Fare) & penanganan missing value
│   ├── 04_test_data_fitur.py        # 4. Ekstraksi fitur test (Age, Fare) & penanganan missing value
│   ├── 05_train_label.py            # 5. Ekstraksi label kelas target pelatihan (Survived)
│   ├── 06_test_label.py             # 6. Ekstraksi label kelas target pengujian (titanic_testlabel.csv)
│   ├── 07_normalisasi_train_minmax.py # 7. Normalisasi Min-Max (0 - 1) data latih & simpan min/max
│   ├── 08_normalisasi_test_minmax.py  # 8. Normalisasi Min-Max (0 - 1) data uji menggunakan parameter train
│   ├── 09_knn_classification.py     # 9. Pemodelan k-NN (k = 1 s.d. 15, weights='distance')
│   ├── 10_precision_error_ratio.py  # 10. Perhitungan Precision Ratio (Akurasi) & Error Ratio
│   └── Uji_Coba.py                  # Skrip gabungan / pipeline lengkap seluruh tahapan
├── materi/
│   └── 4. Klasifikasi k-NN (Teori).pdf
└── README.md
```

---

## 📋 Ringkasan Pengerjaan Soal

| No | File Jawaban | Tugas / Tahapan | Penjelasan Singkat |
|:--:|:---|:---|:---|
| **1** | [`01_load_dataset.py`](jawaban/01_load_dataset.py) | **Load Dataset Train** | Membaca `titanic.csv` menggunakan `pandas` dan menampilkan sampel 5 data teratas serta ringkasan informasi kolom. |
| **2** | [`02_load_test_dataset.py`](jawaban/02_load_test_dataset.py) | **Load Dataset Test** | Membaca `titanic_test.csv` sebagai data uji untuk evaluasi model. |
| **3** | [`03_train_data_fitur.py`](jawaban/03_train_data_fitur.py) | **Fitur Train & Missing Value** | Mengambil fitur `Age` dan `Fare` dari train dataset, mencatat posisi indeks yang missing (*NaN*), lalu membuang baris tersebut. |
| **4** | [`04_test_data_fitur.py`](jawaban/04_test_data_fitur.py) | **Fitur Test & Missing Value** | Mengambil fitur `Age` dan `Fare` dari test dataset, mencatat posisi missing value, dan membersihkannya. |
| **5** | [`05_train_label.py`](jawaban/05_train_label.py) | **Label Target Train** | Mengambil kolom target `Survived` pada dataset pelatihan sesuai baris yang bersih dari missing value. |
| **6** | [`06_test_label.py`](jawaban/06_test_label.py) | **Label Target Test** | Mengambil label target dari `titanic_testlabel.csv` dan membuang baris yang memiliki missing value pada fitur test. |
| **7** | [`07_normalisasi_train_minmax.py`](jawaban/07_normalisasi_train_minmax.py) | **Normalisasi Min-Max Train** | Mentransformasikan fitur `train_data` ke rentang $[0, 1]$ dan mencatat nilai minimum serta maksimum masing-masing fitur. |
| **8** | [`08_normalisasi_test_minmax.py`](jawaban/08_normalisasi_test_minmax.py) | **Normalisasi Min-Max Test** | Menormalkan `test_data` ke rentang $[0, 1]$ menggunakan nilai acuan minimum dan maksimum dari data latih (*Langkah 7*). |
| **9** | [`09_knn_classification.py`](jawaban/09_knn_classification.py) | **Klasifikasi k-NN** | Membangun model `KNeighborsClassifier(weights='distance')` untuk variasi nilai $k = 1$ hingga $15$ dan memprediksi data uji. |
| **10** | [`10_precision_error_ratio.py`](jawaban/10_precision_error_ratio.py) | **Evaluasi Rasio & Error** | Menghitung rasio presisi/akurasi (`precision_ratio`) dan tingkat kesalahan (`error_ratio = 1 - precision_ratio`) untuk $k = 1$ s.d. $15$. |

---

## 📐 Metode & Rumus

### 1. Normalisasi Min-Max (0 - 1)
Mengubah skala nilai fitur agar berada di dalam interval tertutup $[0, 1]$:
$$x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$
*Catatan:* Pada data uji (*test data*), nilai $x_{\min}$ dan $x_{\max}$ yang digunakan wajib berasal dari data latih (*train data*) untuk mencegah kebocoran data (*data leakage*).

### 2. Klasifikasi k-NN dengan Pembobotan Jarak (*Distance Weighted*)
Dalam k-NN dengan `weights='distance'`, tetangga yang berada lebih dekat ke titik data uji diberikan bobot voting yang lebih besar, biasanya berbanding terbalik dengan jarak Euclidean ($w_i = \frac{1}{d_i}$).

### 3. Precision Ratio & Error Ratio
- **Precision Ratio (Akurasi):**
  $$\text{Precision Ratio} = \frac{\text{Jumlah Prediksi Benar}}{\text{Total Data Uji}}$$
- **Error Ratio:**
  $$\text{Error Ratio} = 1 - \text{Precision Ratio}$$

---

## 🚀 Cara Menjalankan

Masuk ke folder jawaban:
```powershell
cd "Tugas 3\jawaban"
```

Jalankan script sesuai kebutuhan:
```powershell
# Menjalankan per langkah:
python 01_load_dataset.py
python 02_load_test_dataset.py
python 03_train_data_fitur.py
python 04_test_data_fitur.py
python 05_train_label.py
python 06_test_label.py
python 07_normalisasi_train_minmax.py
python 08_normalisasi_test_minmax.py
python 09_knn_classification.py
python 10_precision_error_ratio.py

# Atau menjalankan seluruh alur sekaligus:
python Uji_Coba.py
```
