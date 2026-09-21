# Analisis Sentimen Live Chat YouTube MPL ID Season 16 Grand Final

Analisis sentimen penonton Indonesia pada **live chat YouTube Grand Final Mobile Legends Professional League Indonesia (MPL ID) Season 16** menggunakan **IndoBERT** untuk pelabelan otomatis dan **Support Vector Machine (SVM)** untuk klasifikasi.

Proyek akhir mata kuliah **Pemrosesan Teks**, Program Studi S1 Sains Data, Universitas Negeri Surabaya (Kelas 2024E, Kelompok 4).

---

## Latar Belakang

Mobile Legends: Bang Bang adalah salah satu game paling populer di Indonesia, dan MPL ID menjadi turnamen esports yang sangat ramai ditonton. Selama siaran langsung, ribuan komentar mengalir lewat live chat dan mengekspresikan emosi penonton secara spontan. Proyek ini memetakan sentimen (positif, negatif, netral) dari komentar tersebut dan membandingkan dua skenario pemodelan SVM untuk melihat mana yang paling baik.

## Data

| Atribut | Keterangan |
|---|---|
| Sumber | Live chat replay YouTube, kanal **MPL Indonesia** |
| Video | 🔴 LIVE \| MPL ID S16 \| Grand Finals \| Bahasa Indonesia |
| Durasi | 6 jam 46 menit 41 detik |
| Publikasi | 2 November 2025 |
| Metode pengambilan | Scraping dengan library [`pytchat`](https://github.com/taizan-hokuto/pytchat) |
| Data mentah | 10.000 pesan (`timestamp`, `username`, `message`) |
| Setelah preprocessing | ± 4.600 pesan (banyak duplikat akibat spam) |

## Alur Penelitian

```
Scraping Live Chat → Preprocessing → Labeling (IndoBERT) → Visualisasi
→ Pemodelan SVM → Perbandingan Model → Interpretasi
```

### 1. Preprocessing
- Case folding (lowercase)
- Hapus nilai null dan duplikat (banyak pesan spam)
- Hapus URL, mention, hashtag, angka, karakter khusus/emoji, dan huruf berulang
- **Normalisasi slang** dengan kamus manual (mis. `gg`, `ml`, `mlbb`, `krn`, `utk`, `gak`)
- **Stopword removal** dengan Sastrawi (kata negasi/penting seperti `tidak`, `jangan`, `belum` dipertahankan)
- Stemming (Sastrawi) dan tokenisasi

### 2. Pelabelan Otomatis (IndoBERT / RoBERTa)
Label sentimen dihasilkan oleh dua model pretrained dari Hugging Face:

| Model | Negatif | Netral | Positif |
|---|---|---|---|
| `w11wo/indonesian-roberta-base-sentiment-classifier` | 2.212 | 1.122 | 1.270 |
| `mdhugol/indonesia-bert-sentiment-classification` | 1.934 | 1.638 | 1.032 |

Kedua model menunjukkan sentimen **negatif mendominasi**.

### 3. Pemodelan SVM

| | SVM 1 | SVM 2 |
|---|---|---|
| Label dari | `w11wo/indonesian-roberta-...` | `mdhugol/indonesia-bert-...` |
| Fitur | TF-IDF unigram | TF-IDF unigram + bigram (`max_features=5000`) |
| Kernel | RBF (`C=1.0`, `gamma='scale'`) | Linear (`C=1.0`) |
| Penanganan imbalance | – | `class_weight='balanced'`, stratified split |
| Split | 80:20, `random_state=42` | 80:20, `random_state=42` |

## Hasil

| Metrik | SVM 1 (RBF) | SVM 2 (Linear + bigram) |
|---|---|---|
| Akurasi | 66,12% | **73,29%** |
| Macro F1 | 0,62 | 0,73 |
| F1 kelas netral | 0,49 | 0,73 |

SVM 2 lebih seimbang antar kelas, terutama pada kelas netral yang paling sulit dideteksi, dan tidak terlalu condong ke kelas negatif seperti SVM 1.

> **Catatan metodologis:** kedua SVM dilatih pada label dari model IndoBERT yang **berbeda**, dengan konfigurasi fitur dan kernel yang juga berbeda. Selisih akurasi 7,29% karena itu mencerminkan gabungan semua perbedaan tersebut, bukan efek satu faktor saja. Label juga berupa *pseudo-label* dari model pretrained (bukan anotasi manual), sehingga akurasi SVM mengukur kesesuaian dengan label model IndoBERT, bukan dengan sentimen sebenarnya.

## Struktur Repo

```
.
├── mlbb-mpl-s16-livechat-sentiment-indobert-svm_data processing.ipynb       # Import data + preprocessing (cleaning, normalisasi, stopword, stemming)
├── mlbb-mpl-s16-livechat-sentiment-indobert-svm_Labeling_dan_Modeling.ipynb  # Labeling IndoBERT, visualisasi, pemodelan & perbandingan SVM
└── README.md
```

## Cara Menjalankan

1. **Clone repo**
```bash
   git clone https://github.com/<username>/mpl-id-s16-sentiment-analysis.git
   cd mpl-id-s16-sentiment-analysis
```

2. **Install dependensi**
```bash
   pip install pandas nltk sastrawi scikit-learn transformers torch wordcloud matplotlib seaborn jupyter
   # opsional, hanya jika ingin scraping ulang:
   pip install pytchat
```

3. **Siapkan data**: letakkan `livechatyt.csv` di folder `data/`, atau scraping ulang dengan `pytchat` memakai video ID `kxOQKzUif4I` (target 10.000 pesan).

4. **Jalankan notebook secara berurutan**
   1. `mlbb-mpl-s16-livechat-sentiment-indobert-svm_data processing.ipynb` menghasilkan `hasil_stemming.csv`
   2. `mlbb-mpl-s16-livechat-sentiment-indobert-svm_Labeling_dan_Modeling.ipynb` untuk labeling, visualisasi, dan SVM

   Sesuaikan path file CSV di notebook dengan lokasi file di komputermu. Model Hugging Face akan terunduh otomatis saat pertama kali dijalankan.

## Temuan Utama

- Sentimen **negatif dominan** pada kedua model pelabelan, mencerminkan respons emosional dan kritis penonton selama pertandingan.
- Sekitar **54%** data mentah terbuang saat penghapusan duplikat karena banyaknya spam di live chat.
- Word cloud didominasi nama tim dan istilah pertandingan (mis. *alter ego*, *onic*, *rrq*, *evos*, *juara*, *menang*).
- TF-IDF + bigram, kernel linear, dan `class_weight='balanced'` memberikan hasil lebih seimbang daripada TF-IDF unigram dengan RBF default.

## Saran Pengembangan

- Bandingkan kedua SVM pada **label dan konfigurasi yang sama** agar efek tiap faktor (kernel, bigram, class weight) terukur terpisah
- Validasi dengan **anotasi manual** pada sampel data sebagai *ground truth*
- *Fine-tuning* IndoBERT langsung pada data live chat, dan hyperparameter tuning (`C`, `gamma`) dengan cross-validation
- Analisis sentimen berdasarkan **timeline** pertandingan untuk melihat reaksi penonton pada momen tertentu

## Tim (Kelompok 4)

| NIM | Nama |
|---|---|
| 24031554040 | Bima Setia Sugiharto |
| 24031554091 | Muhammad Geralldo Agatha Saputra |
| 24031554132 | Moh. Rasya Al Khalifi |

Program Studi S1 Sains Data, Fakultas Matematika dan Ilmu Pengetahuan Alam, Universitas Negeri Surabaya

## Referensi

- Hadin, L. A. S. H. W., & Nurjanah, S. (2024). Analisis Sentimen Komentar Netizen pada Siaran Langsung YouTube RRQ Hoshi vs EVOS Legends MPL Season 12 Menggunakan Metode Naïve Bayes. *Jurnal Sistem Informasi Indonesia, 4*(2).
- Nugroho, T. S. E., Utomo, A. S. A., & Prasetio, B. H. (2024). Analisis Sentimen Ulasan Pengguna Aplikasi Mobile Legends pada Google Play Store Menggunakan IndoBERT. *Jurnal SIFO SAGA, 15*(1).
- Tcka, D. M. G., et al. (2024). Understanding E-sports audience motivation: An analysis of live streaming chat. *Journal of Documentation, 80*(3).

## Lisensi

Proyek ini dibuat untuk keperluan akademik. Data live chat berasal dari siaran publik kanal MPL Indonesia di YouTube.
