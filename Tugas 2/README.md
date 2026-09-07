# 🚢 Praktikum Data Preprocessing: Imputasi Missing Value & Normalisasi Data - Titanic Dataset

Repositori ini berisi pengerjaan tugas dan latihan **Pertemuan 3 Mata Kuliah Web Data Mining**, yang berfokus pada tahapan *data preprocessing*: penanganan nilai kosong (*missing value imputation*) berbasis grup kelas, serta teknik-teknik normalisasi data numerik (**Min-Max**, **Z-Score**, dan **Sigmoida**) menggunakan library **Pandas** dan **NumPy** pada dataset penumpang kapal Titanic.

---

## 📁 Struktur Direktori

```text
Tugas 2/
├── data/
│   └── titanic.csv                  # Dataset mentah Titanic
├── jawaban/
│   ├── 01_load_dataset.py           # 1. Membaca & menampilkan dataset
│   ├── 02_rows_cols.py              # 2. Menghitung jumlah baris dan kolom (rows, cols)
│   ├── 03_data_fitur.py             # 3. Mengambil subset fitur (Age, Fare)
│   ├── 04_class_target.py           # 4. Mengambil kolom target kelas (Survived)
│   ├── 05_missing_value_age.py      # 5. Imputasi missing value Age dengan mean per class
│   ├── 06_normalisasi_min_max.py    # 6. Normalisasi fitur dengan metode Min-Max (0 - 1)
│   ├── 07_normalisasi_z_score.py    # 7. Normalisasi fitur dengan metode Z-Score
│   └── 08_normalisasi_sigmoida.py   # 8. Normalisasi fitur dengan metode Sigmoida
├── materi/
│   └── 3. Data preprocessing (Data Normalization) (prak).pdf
└── README.md
```

---

## 📋 Ringkasan Pengerjaan Soal

| No | File Jawaban | Tugas / Soal | Penjelasan Singkat |
|:--:|:---|:---|:---|
| **1** | [`01_load_dataset.py`](jawaban/01_load_dataset.py) | **Load Dataset** | Membaca file `titanic.csv` menggunakan `pd.read_csv()` dan menampilkan seluruh data serta sampel 5 data teratas (`.head()`). |
| **2** | [`02_rows_cols.py`](jawaban/02_rows_cols.py) | **Dimensi Dataset** | Mengekstrak jumlah baris dan kolom dataset via `dataset.shape` (Hasil: **891 baris** dan **12 kolom**). |
| **3** | [`03_data_fitur.py`](jawaban/03_data_fitur.py) | **Ekstraksi Fitur** | Mengisolasi kolom atribut numerik `data = dataset[['Age', 'Fare']]` untuk tahap pemrosesan data. |
| **4** | [`04_class_target.py`](jawaban/04_class_target.py) | **Ekstraksi Kelas** | Mengambil label kelas target `target_class = dataset['Survived']` (`0` = Tidak Selamat, `1` = Selamat). |
| **5** | [`05_missing_value_age.py`](jawaban/05_missing_value_age.py) | **Imputasi Missing Value** | Mengisi nilai kosong (*NaN*) pada kolom `Age` menggunakan rata-rata (*mean*) umur dari masing-masing kelompok kelas keselamatan (`Survived`). |
| **6** | [`06_normalisasi_min_max.py`](jawaban/06_normalisasi_min_max.py) | **Normalisasi Min-Max** | Mentransformasikan fitur `Age` dan `Fare` ke rentang skala `[0, 1]`. |
| **7** | [`07_normalisasi_z_score.py`](jawaban/07_normalisasi_z_score.py) | **Normalisasi Z-Score** | Standarisasi fitur agar memiliki rata-rata (*mean*) = 0 dan standar deviasi (*std*) = 1. |
| **8** | [`08_normalisasi_sigmoida.py`](jawaban/08_normalisasi_sigmoida.py) | **Normalisasi Sigmoida** | Memetakan nilai data hasil transformasi Z-Score ke fungsi logistik non-linear bernilai antara `(0, 1)`. |

---

## 📐 Penjelasan Metode & Rumus yang Digunakan

### 1. Imputasi Missing Value Berdasarkan Kelas
Alih-alih menggunakan mean global seluruh populasi, nilai kosong pada `Age` diisi dengan rata-rata umur kelompok penumpangnya:
$$\text{Age}_{new} = \begin{cases} \mu_{\text{Age}\mid\text{Survived}=0}, & \text{jika Survived}=0 \text{ dan Age}=\text{NaN} \\ \mu_{\text{Age}\mid\text{Survived}=1}, & \text{jika Survived}=1 \text{ dan Age}=\text{NaN} \end{cases}$$

Implementasi kode:
```python
dataset['Age'] = dataset.groupby('Survived')['Age'].transform(lambda x: x.fillna(x.mean()))
```

### 2. Normalisasi Min-Max (0 - 1)
Mengubah skala nilai fitur sehingga berada dalam interval tertutup `[0, 1]`.
$$x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

- **Kelebihan:** Mempertahankan hubungan proporsional asli dan memastikan tidak ada fitur bernilai dominan karena perbedaan satuan.
- **Sensitivitas:** Rentan terhadap pencilan (*outliers*).

### 3. Normalisasi Z-Score (Standarisasi)
Mengubah distribusi data sehingga berpusat di sekitar 0 dengan variansi unit.
$$z = \frac{x - \mu}{\sigma}$$
*Keterangan:* $\mu$ = rata-rata (*mean*), $\sigma$ = standar deviasi (*standard deviation*).

- **Kelebihan:** Sangat baik menangani data yang berdistribusi normal dan tidak terikat batas interval tetap.

### 4. Normalisasi Sigmoida (Sigmoidal / Logistic)
Menerapkan fungsi aktivasi sigmoid pada nilai standar $z$:
$$S(z) = \frac{1}{1 + e^{-z}}$$

- **Kelebihan:** Mengurangi pengaruh nilai ekstrem (*outliers*) secara halus (*smooth non-linear scaling*) ke rentang `(0, 1)`.

---

## 🚀 Cara Menjalankan

1. **Pastikan library yang dibutuhkan telah terpasang:**
   ```bash
   pip install pandas numpy
   ```

2. **Masuk ke direktori folder jawaban:**
   ```powershell
   cd "Tugas 2\jawaban"
   ```

3. **Jalankan script sesuai nomor yang diinginkan:**
   ```powershell
   # 1. Menampilkan dataset
   python 01_load_dataset.py

   # 2. Cek baris dan kolom
   python 02_rows_cols.py

   # 3. Fitur Age dan Fare
   python 03_data_fitur.py

   # 4. Kelas Survived
   python 04_class_target.py

   # 5. Imputasi Missing Value
   python 05_missing_value_age.py

   # 6. Normalisasi Min-Max
   python 06_normalisasi_min_max.py

   # 7. Normalisasi Z-Score
   python 07_normalisasi_z_score.py

   # 8. Normalisasi Sigmoida
   python 08_normalisasi_sigmoida.py
   ```

---

## 💡 Kesimpulan
- **Pengisian Nilai Kosong Berdasarkan Kelompok:** Mengisi umur yang kosong menggunakan rata-rata kelompoknya masing-masing (kelompok selamat dan tidak selamat) jauh lebih tepat dan masuk akal dibandingkan jika disamaratakan untuk seluruh penumpang.
- **Perbedaan Metode Normalisasi:**
  - **Min-Max:** Mengubah skala data agar pas di rentang angka 0 sampai 1, cocok agar fitur dengan angka besar tidak mendominasi fitur lain.
  - **Z-Score:** Menyelaraskan data agar berpusat di angka 0, berguna untuk melihat seberapa jauh suatu nilai berada di atas atau di bawah rata-rata.
  - **Sigmoida:** Mengubah data menjadi rentang 0 sampai 1 secara melengkung halus, sangat bagus agar nilai yang terlalu ekstrem (terlalu tinggi atau terlalu rendah) tidak merusak pola data.
