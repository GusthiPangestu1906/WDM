# 🌐 Web Data Mining (WDM)

Repositori ini berisi kumpulan tugas, praktikum, dan materi mata kuliah **Web Data Mining** dengan studi kasus analisis dan *preprocessing* data menggunakan **Python**, **Pandas**, dan **NumPy**.

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

---

## 🚀 Persyaratan & Instalasi

Pastikan telah menginstall library Python yang dibutuhkan:

```bash
pip install pandas numpy matplotlib seaborn
```
