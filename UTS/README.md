# 🚢 UTS - Klasifikasi untuk Imputasi (Titanic Dataset)

Repositori ini berisi pengerjaan **UTS Mata Kuliah Web Data Mining** berdasarkan modul penugasan *Knowledge Engineering (knoWing) Research Group - Politeknik Elektronika Negeri Surabaya (PENS)*.

Tugas ini berfokus pada dua fase utama:
1. **Bagian 1 (Langkah 1 – 9):** Imputasi nilai kosong (*missing values*) pada atribut `Age` menggunakan algoritma **3-NN** (k-Nearest Neighbors, $k=3$) pada dataset pelatihan (`titanic.csv`).
2. **Bagian 2 (Langkah 10 – 19):** Klasifikasi tingkat keselamatan penumpang (`Survived`) menggunakan model **3-NN** pada dataset pengujian (`titanic_test.csv` & `titanic_testlabel.csv`), serta perhitungan **Error Ratio** dan **Akurasi**.

---

## 📁 Struktur File & Direktori

```text
UTS/
├── data/
│   ├── titanic.csv                     # Dataset Titanic Latih (891 baris, 12 kolom)
│   ├── titanic_test.csv                # Dataset Titanic Uji (418 baris, 11 kolom)
│   └── titanic_testlabel.csv           # Label Uji Survived (418 baris, 2 kolom)
├── jawaban/
│   ├── data_loader.py                  # Helper khusus pemuatan dataset (load data)
│   ├── 01_load_dataset.py              # Langkah 1: dataset <- titanic.csv
│   ├── 02_ekstraksi_fitur_data.py      # Langkah 2: data <- fitur (Sex, Age, Pclass, Fare, Survived)
│   ├── 03_train_data_imputasi.py       # Langkah 3: train_data <- baris Age != null
│   ├── 04_train_label_imputasi.py      # Langkah 4: train_label <- target Age != null
│   ├── 05_test_data_imputasi.py        # Langkah 5: test_data <- baris Age == null
│   ├── 06_normalisasi_train_imputasi.py# Langkah 6: normalisasi Min-Max 0-1 train_data
│   ├── 07_normalisasi_test_imputasi.py # Langkah 7: normalisasi Min-Max 0-1 test_data
│   ├── 08_knn_imputasi_age.py          # Langkah 8: class_result <- klasifikasi 3-NN untuk Age
│   ├── 09_pengisian_missing_value_age.py # Langkah 9: pengisian nilai Age kosong
│   ├── 10_load_test_dataset.py         # Langkah 10: test_dataset <- titanic_test.csv & titanic_testlabel.csv
│   ├── 11_train_data_klasifikasi.py    # Langkah 11: train_data <- fitur (Sex, Age, Pclass, Fare)
│   ├── 12_train_label_klasifikasi.py   # Langkah 12: train_label <- fitur Survived
│   ├── 13_test_data_klasifikasi.py     # Langkah 13: test_data <- buang missing values
│   ├── 14_test_label_klasifikasi.py    # Langkah 14: test_label <- target Survived sesuai no 13
│   ├── 15_normalisasi_train_klasifikasi.py # Langkah 15: normalisasi Min-Max 0-1 train_data
│   ├── 16_normalisasi_test_klasifikasi.py  # Langkah 16: normalisasi Min-Max 0-1 test_data
│   ├── 17_knn_klasifikasi_survived.py  # Langkah 17: class_result <- klasifikasi 3-NN Survived
│   ├── 18_hitung_jumlah_error.py       # Langkah 18: hitung jumlah kesalahan prediksi
│   └── 19_hitung_error_ratio.py        # Langkah 19: perhitungan error ratio (%) dan akurasi (%)
└── README.md
```

---

## 📋 Rincian 19 Langkah Pengerjaan

| No | File Skrip | Deskripsi Langkah |
| :-: | :--- | :--- |
| **1** | [`01_load_dataset.py`](jawaban/01_load_dataset.py) | Memuat dataset utama `titanic.csv` (891 baris). |
| **2** | [`02_ekstraksi_fitur_data.py`](jawaban/02_ekstraksi_fitur_data.py) | Mengambil kolom `Sex`, `Age`, `Pclass`, `Fare`, dan `Survived`, serta mengonversi gender ke numerik (`female: 0`, `male: 1`). |
| **3** | [`03_train_data_imputasi.py`](jawaban/03_train_data_imputasi.py) | Mengambil fitur `Sex`, `Pclass`, `Fare`, `Survived` pada baris yang memiliki nilai `Age != null` (714 baris). |
| **4** | [`04_train_label_imputasi.py`](jawaban/04_train_label_imputasi.py) | Mengambil label target `Age` dari baris yang tidak null (714 sampel). |
| **5** | [`05_test_data_imputasi.py`](jawaban/05_test_data_imputasi.py) | Mengambil fitur `Sex`, `Pclass`, `Fare`, `Survived` pada baris yang memiliki `Age == null` (177 baris missing). |
| **6** | [`06_normalisasi_train_imputasi.py`](jawaban/06_normalisasi_train_imputasi.py) | Normalisasi Min-Max 0–1 pada `train_data` imputasi dan mencatat nilai $min$ serta $max$ tiap atribut. |
| **7** | [`07_normalisasi_test_imputasi.py`](jawaban/07_normalisasi_test_imputasi.py) | Normalisasi Min-Max 0–1 pada `test_data` imputasi menggunakan parameter dari Langkah 6. |
| **8** | [`08_knn_imputasi_age.py`](jawaban/08_knn_imputasi_age.py) | Melatih model 3-NN ($k=3$) untuk memprediksi kelas umur (`class_result`) pada 177 data uji. |
| **9** | [`09_pengisian_missing_value_age.py`](jawaban/09_pengisian_missing_value_age.py) | Mengisi baris kosong pada kolom `Age` dengan nilai `class_result` sehingga dataset menjadi lengkap (0 missing). |
| **10** | [`10_load_test_dataset.py`](jawaban/10_load_test_dataset.py) | Membaca dataset uji langsung dari `titanic_test.csv` (418 baris) dan `titanic_testlabel.csv` (418 baris). |
| **11** | [`11_train_data_klasifikasi.py`](jawaban/11_train_data_klasifikasi.py) | Mengambil fitur `Sex`, `Age` (hasil imputasi), `Pclass`, dan `Fare` dari dataset latih (891 baris). |
| **12** | [`12_train_label_klasifikasi.py`](jawaban/12_train_label_klasifikasi.py) | Mengambil label keselamatan `Survived` dari dataset latih (891 baris). |
| **13** | [`13_test_data_klasifikasi.py`](jawaban/13_test_data_klasifikasi.py) | Mengambil fitur pada `test_dataset` dan membuang baris yang memiliki nilai missing (tersisa 331 baris bersih). |
| **14** | [`14_test_label_klasifikasi.py`](jawaban/14_test_label_klasifikasi.py) | Mengambil label aktual `Survived` dari `titanic_testlabel.csv` dengan indeks tersinkronisasi (331 sampel). |
| **15** | [`15_normalisasi_train_klasifikasi.py`](jawaban/15_normalisasi_train_klasifikasi.py) | Normalisasi Min-Max 0–1 pada `train_data` klasifikasi dan mencatat nilai $min$ serta $max$. |
| **16** | [`16_normalisasi_test_klasifikasi.py`](jawaban/16_normalisasi_test_klasifikasi.py) | Normalisasi Min-Max 0–1 pada `test_data` klasifikasi menggunakan parameter dari Langkah 15. |
| **17** | [`17_knn_klasifikasi_survived.py`](jawaban/17_knn_klasifikasi_survived.py) | Melatih model 3-NN ($k=3$) dan memprediksi status keselamatan `Survived` pada data uji. |
| **18** | [`18_hitung_jumlah_error.py`](jawaban/18_hitung_jumlah_error.py) | Membandingkan `class_result` dengan `test_label` aktual dan menghitung total prediksi salah ($error$). |
| **19** | [`19_hitung_error_ratio.py`](jawaban/19_hitung_error_ratio.py) | Menghitung persentase kesalahan ($Error\ Ratio$) dan tingkat akurasi model. |

---

## 📊 Hasil Evaluasi Akhir

| Metrik Evaluasi | Nilai | Keterangan |
| :--- | :---: | :--- |
| **Total Sampel Uji Bersih** | **331** | Data uji setelah eliminasi baris missing (87 baris tereliminasi) |
| **Jumlah Prediksi Benar** | **271** | Prediksi 3-NN sesuai dengan label aktual |
| **Jumlah Prediksi Salah (Error)** | **60** | Selisih antara prediksi dan label aktual |
| **Error Ratio (%)** | **18.13%** | $\frac{60}{331} \times 100\%$ |
| **Akurasi (%)** | **81.87%** | $100\% - 18.13\%$ |

---

## 🚀 Cara Menjalankan

### Cara Menjalankan Skrip:
```bash
# Contoh: Langkah 1
python UTS/jawaban/01_load_dataset.py

# Contoh: Langkah 10 (Load Data Uji CSV Langsung)
python UTS/jawaban/10_load_test_dataset.py

# Contoh: Langkah 19 (Hasil Evaluasi Error & Akurasi Akhir)
python UTS/jawaban/19_hitung_error_ratio.py
```
