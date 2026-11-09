# Credit Risk Prediction Model

Final task **Project-Based Internship ID/X Partners x Rakamin Academy (VIX) — Data Scientist**.

Project ini membangun model machine learning untuk memprediksi risiko kredit pada perusahaan multifinance, dari pemahaman data, EDA, data preparation, hingga pemodelan dan evaluasi. Dua algoritma dibandingkan secara setara dengan cross-validation: **Logistic Regression** (wajib) dan **HistGradientBoosting**.

## Ringkasan Hasil

| | Logistic Regression | HistGradientBoosting |
|---|---|---|
| AUC cross-validation (3-fold) | 0,7120 | 0,7134 |
| AUC train | 0,7133 | 0,7395 |
| **AUC test** | **0,7147** | **0,7197** |
| PR-AUC test (baseline acak 0,214) | 0,3997 | 0,4068 |
| KS test | 0,3161 | 0,3221 |
| Gini test | 0,4295 | 0,4393 |

- AUC sekitar 0,72 adalah angka yang **wajar** untuk dataset ini karena kolom *data leakage* sudah dibuang. AUC yang mendekati 1 justru patut dicurigai.
- Selisih AUC train dan test kecil, sehingga tidak ada indikasi overfitting yang berarti.
- Kedua model hampir setara. HistGradientBoosting unggul tipis, sedangkan Logistic Regression lebih mudah dijelaskan.

**Kemampuan memisahkan risiko (test set, HistGradientBoosting):**

- Bad rate desil skor teratas **48,1%** vs desil terbawah **4,3%** (sekitar 11 kali lipat).
- Tiga desil teratas menangkap **54%** dari seluruh pinjaman bad.
- Contoh kebijakan: dengan cutoff PD 0,20, perusahaan menyetujui 54,5% pemohon, bad rate pinjaman yang disetujui turun dari 21,4% menjadi **11,6%**, dan 70,5% pinjaman bad tertangkap.

![ROC dan PR curve](figures/09_roc_pr_curves.png)

![Bad rate per desil dan kalibrasi](figures/11_decile_calibration.png)

## Latar Belakang

Client adalah perusahaan pemberi pinjaman (multifinance) yang ingin menilai dan mengelola risiko kredit lebih akurat untuk mengurangi potensi kerugian. Tugasnya adalah memprediksi apakah seorang peminjam akan menjadi pinjaman bermasalah (*bad*) atau lancar (*good*).

## Dataset

- **Sumber:** data pinjaman Lending Club periode 2007-2014 (`loan_data_2007_2014.csv`) beserta kamus data (`LCDataDictionary.xlsx`), disediakan oleh program.
- **Ukuran:** 466.285 baris x 75 kolom, tanpa duplikat.
- Dataset **tidak disertakan** di repository karena ukurannya besar (lihat `.gitignore`).

### Penentuan label

Dataset tidak memiliki kolom label. Label dibuat dari kolom performa kredit `loan_status`:

| Label | loan_status |
|---|---|
| **Bad (1)** | Charged Off, Default, Late (31-120 days), Does not meet the credit policy: Charged Off |
| **Good (0)** | Fully Paid, Does not meet the credit policy: Fully Paid |
| **Dikeluarkan** | Current, In Grace Period, Late (16-30 days) — hasil akhir pinjaman belum diketahui |

Hasilnya **237.695 pinjaman berlabel** dengan bad rate **21,4%** (data imbalanced, sehingga metrik utama adalah AUC, PR-AUC, dan KS, bukan accuracy).

![Distribusi label](figures/02_label_distribution.png)

## Metodologi

1. **Data Understanding** — struktur dataset, tipe data, statistik deskriptif, dan pemeriksaan kualitas data (missing value, duplikat, outlier).
2. **EDA** — analisis univariat, bivariat, korelasi, dan tren per tahun penerbitan.
3. **Data Preparation**
   - Membuang kolom *leakage* (13 kolom yang baru terisi setelah pinjaman berjalan, mis. `total_pymnt`, `recoveries`), kolom ID/teks bebas, kolom redundan, kolom missing di atas 50%, dan kolom konstan. Dari 75 kolom tersisa 29 fitur.
   - Feature engineering: `term_months`, `emp_length_yrs`, `credit_history_yrs`, `loan_to_income`, `installment_to_income`, transformasi `log1p` pada fitur yang sangat skewed.
   - Split train/test 80:20 secara *stratified* **sebelum** preprocessing.
   - Preprocessing di dalam `Pipeline` (tanpa kebocoran ke test maupun fold validasi): winsorizing persentil 0,5/99,5, imputasi median + indikator missing, scaling, dan one-hot encoding (105 fitur).
4. **Data Modelling** — tuning dengan 3-fold stratified CV (metrik ROC-AUC) pada sampel 60.000 baris, lalu model terbaik dilatih ulang pada seluruh data latih.
   - Logistic Regression: `GridSearchCV` (6 kombinasi). Terbaik: `C = 0,01`, `class_weight = balanced`.
   - HistGradientBoosting: `RandomizedSearchCV` (8 kombinasi). Terbaik: `learning_rate = 0,05`, `max_leaf_nodes = 31`, `min_samples_leaf = 100`, `l2_regularization = 10`, `max_iter = 200`.
5. **Evaluation** — evaluasi pada test set terpisah: AUC, PR-AUC, KS, Gini, pemeriksaan overfitting, confusion matrix, tabel cutoff, tabel desil risiko, kurva kalibrasi, dan feature importance.

## Temuan Utama

- **Grade** adalah pembeda risiko yang paling jelas: bad rate naik dari 7,3% (grade A) menjadi 47,0% (grade G).
- Tenor **60 bulan** memiliki bad rate 34,9% vs 17,7% untuk 36 bulan. Tujuan pinjaman `small_business` berisiko tinggi (31,7%).
- Faktor pendorong risiko terbesar menurut permutation importance adalah `int_rate`, diikuti `term_months`, `annual_inc`, `loan_to_income`, dan `dti`.
- Probabilitas dari HistGradientBoosting terkalibrasi dengan baik (rata-rata prediksi sejalan dengan bad rate aktual tiap desil). Sebaliknya, probabilitas Logistic Regression dengan `class_weight=balanced` **terlalu tinggi** (overestimate). Kemampuan *meranking* risikonya tetap baik, tetapi angkanya perlu dikalibrasi ulang sebelum dibaca sebagai probabilitas gagal bayar.

![Feature importance](figures/12_feature_importance.png)

## Rekomendasi Bisnis

1. Gunakan skor probabilitas gagal bayar (PD) sebagai **alat bantu keputusan**, bukan keputusan otomatis tunggal.
2. Terapkan kebijakan bertingkat: PD rendah disetujui, menengah direview manual, tinggi ditolak atau diberi bunga lebih tinggi.
3. Tentukan cutoff dari trade-off antara approval rate dan bad rate, serta biaya kredit macet.
4. Pantau drift dan kalibrasi model secara berkala.

## Keterbatasan

- Label hanya berasal dari pinjaman yang sudah selesai, sehingga ada **bias kematangan**. Bad rate pinjaman 2013-2014 tampak lebih tinggi karena pinjaman yang dilunasi penuh baru selesai setelah 2014. Tren per tahun tidak boleh dibaca sebagai penurunan kualitas kredit.
- Data hanya berisi pinjaman yang disetujui (tidak ada *reject inference*).
- Validasi memakai split acak, bukan *out-of-time*.
- Pencarian hyperparameter dilakukan pada sampel 60.000 baris untuk efisiensi.

## Struktur Repository

```
.
├── README.md
├── .gitignore
├── ID-X_credit_risk.ipynb     # notebook lengkap (kode, penjelasan, dan output)
├── credit_risk_model.py       # versi script (isi sama dengan notebook)
├── figures/                   # grafik hasil EDA dan evaluasi
└── presentation/              # (opsional) slide final submission
```

## Cara Menjalankan

1. Clone repository ini.
2. Taruh `loan_data_2007_2014.csv` di folder yang sama dengan notebook/script (atau tentukan lokasinya lewat environment variable `LOAN_DATA`).
3. Install dependency:

   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```

4. Jalankan notebook `ID-X_credit_risk.ipynb` dari awal sampai akhir (*Run All*), atau jalankan script:

   ```bash
   python credit_risk_model.py
   # atau dengan lokasi data kustom:
   LOAN_DATA=/path/ke/loan_data_2007_2014.csv python credit_risk_model.py
   ```

Grafik akan tersimpan di folder `figures/` (dapat diubah lewat environment variable `FIG_DIR`). Seluruh proses memakan waktu sekitar 3-4 menit pada satu core CPU.

**Versi yang diuji:** Python 3.12, pandas 3.0, scikit-learn 1.9, matplotlib 3.11. Kode memerlukan scikit-learn 1.2 atau lebih baru (`OneHotEncoder` dengan `sparse_output` dan `min_frequency`).

## Penulis

**Muhammad Husain As Shiddiqy** — [LinkedIn](https://www.linkedin.com/in/muhusainas/) | [muhusain01@gmail.com](mailto:muhusain01@gmail.com)

*Dikerjakan sebagai final task Project-Based Internship ID/X Partners x Rakamin Academy.*
