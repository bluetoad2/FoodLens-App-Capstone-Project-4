# FoodLens App | Object Detection with YOLOv12 Model

**FoodLens** adalah aplikasi web berbasis Streamlit yang menggunakan AI Object Detection (YOLOv12) untuk mendeteksi kalori dan nilai gizi makanan dari foto dan memberikan analisis gizi secara cepat. Aplikasi ini bertujuan untuk membantu user melacak asupan kalori dan makronutrien, memantau riwayat konsumsi, serta mendapatkan saran gizi dari AI (ChatGPT LLM).

---

## Project Overview
Project ini merupakan Capstone Project Module 4 - AI Engineering, yang berfokus pada penerapan Object Detection (dengan YOLOv12) untuk mendeteksi dan menghitung kalori makanan serta dikemas dalam aplikasi Streamlit yaitu FoodLens App.

Tujuan utama dari project ini adalah untuk menyediakan aplikasi yang praktis bagi user untuk menghitung nilai gizi makanan dengan lebih mudah, hanya dengan menggunakan foto makanan.

Model AI Object Detection untuk makanan di-train dan dibuat di Google Colab dalam bentuk Jupyter Notebook. Untuk penjelasan lengkap cara training model YOLOv12, detail setting parameter, dan hasil evaluasi akurasi model yang digunakan pada aplikasi ini dapat dibuka di file ` Capstone_Project_4_Computer_Vision_Calory_Dataset_Christopher_Daniel_S.ipynb` 

Aplikasi FoodLens lalu dibuat menggunakan **Python** dengan framework **Streamlit**. Untuk fitur rekomendasi dari AI, aplikasi ini mengintegrasikan model **OpenAI (gpt-4o-mini)** melalui **LangChain** untuk memberikan saran gizi tambahan dari foto makanan user.

Seluruh proses development aplikasi dibuat di local environment Python menggunakan VS Code / Streamlit, dan di-deploy ke Streamlit Cloud.

---

## Ringkasan Fitur
### Fitur Utama
- Object Detection (dengan YOLOv12) untuk mendeteksi dan menghitung total kalori makanan dari foto makanan user.

### Fitur Tambahan
#### 1. Analisis Foto Makanan + Detail Tambahan (Makronutrien dan Confidence Level) ✅
  - Unggah satu gambar (`.jpg`, `.png`, `.jpeg`).
  - Menampilkan gambar asli dan gambar hasil deteksi dengan *bounding box* di sekitar makanan.
  - Menampilkan tabel rincian nutrisi (kalori, protein, karbohidrat, lemak) dan Confidence Level deteksi.
  - Menampilkan ringkasan total nutrisi dari semua makanan yang terdeteksi.

#### 2. Rekomendasi + Resep dari AI LLM ✅
  - Tombol "Dapatkan Saran AI & Resep" yang muncul setelah analisis gambar berhasil.
  - Menggunakan model `gpt-4o-mini` dari OpenAI untuk memberikan rekomendasi makanan alternatif yang lebih sehat dan ide resep praktis.
  - Menampilkan estimasi penggunaan token dan biaya untuk setiap permintaan ke API OpenAI.

#### 3. Dashboard History Makanan ✅
  - Halaman khusus untuk melihat riwayat semua makanan yang telah dianalisis dan disimpan.
  - Filter data berdasarkan rentang waktu (7 hari terakhir, 30 hari terakhir, atau semua riwayat).
  - Visualisasi data interaktif menggunakan **Plotly** menunjukkan:
    - Tren asupan kalori harian.
    - Komposisi makronutrien (dalam bentuk diagram lingkaran).
    - Frekuensi makanan yang paling sering dikonsumsi.
  - Tombol untuk menghapus seluruh riwayat jika pengguna ingin memulai dari awal.

#### 3. Batch Analisis Foto Makanan + Laporan (CSV/PDF) ✅
  - Fitur untuk mengunggah beberapa file gambar sekaligus atau dalam bentuk file `.zip`.
  - Menampilkan progress bar selama proses analisis.
  - Hasil dari semua gambar digabungkan dalam satu tabel ringkasan.
  - Opsi untuk mengunduh laporan lengkap dalam format **CSV** atau **PDF**.

---

## Alur Kerja Aplikasi

1.  **Input User**: User mengunggah satu atau beberapa gambar makanan melalui UI Streamlit.
2.  **Deteksi Objek**: Gambar diproses oleh model **YOLOv12** (`best.pt`) yang telah dilatih untuk mengenali berbagai jenis makanan.
3.  **Ekstraksi Informasi**: Nama makanan dan informasi nutrisi (kalori, protein, karbohidrat, lemak) diekstrak dari data yang telah ditentukan sebelumnya.
4.  **Perhitungan & Tampilan**: Aplikasi menghitung dan menampilkan rincian nutrisi per item dan ringkasan totalnya. Aplikasi juga menampilkan gambar asli dengan kotak pembatas yang ditumpangkan pada item yang terdeteksi.
5.  **Saran AI**: Jika pengguna meminta saran, daftar makanan yang terdeteksi akan dikirim ke **API OpenAI** melalui **LangChain** untuk menghasilkan saran diet dan resep yang dipersonalisasi.
6.  **History Makanan**: Jika pengguna memilih untuk menyimpan hasilnya, data akan ditulis ke dalam database **SQLite**.

---

## Struktur File Project

```
Jupyter Notebook - Capstone 4/
│
├── .streamlit/
│   └── secrets.toml        # Tempat menyimpan kunci API OpenAI
│
├── assets/
│   └── logo.png            # Logo aplikasi
│
├── database/
│   └── foodlens.db         # File database SQLite
│
├── models/
│   └── best.pt             # File model deteksi objek YOLO
│
├── FoodLens App.py         # Skrip utama aplikasi Streamlit
├── requirements.txt        # Daftar library yang dibutuhkan
├── conclusion.md           # Catatan kesimpulan proyek
└── README.md               # File README
```

---

## Environment / Secrets

Untuk menggunakan fitur Asisten Gizi AI, Dibuat file `.streamlit/secrets.toml` dan memasukkan API Key OpenAI di dalamnya:

```toml
OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

---

## Dependencies

Daftar library Python yang dibutuhkan untuk menjalankan aplikasi ini:

```
streamlit
opencv-python-headless
numpy
pandas
Pillow
plotly
langchain
langchain-openai
reportlab
ultralytics
supervision
```

---

## Deployment (Streamlit Cloud)
Demo Aplikasi FoodLens dapat diakses di link berikut : https://foodlens-detection.streamlit.app/

---

## Cara Penggunaan Aplikasi
1.  **Pilih Halaman:** Gunakan navigasi di atas untuk memilih mode analisis.
2.  **Atur Deteksi:** Sesuaikan 'Confidence Level' dan 'IoU' jika perlu (Semakin tinggi, semakin strict).
3.  **Upload Foto:** Pilih foto makanan kamu yang ingin dianalisis.
4.  **Lihat Hasil:** Periksa tabel nutrisi dan total kalori.
5.  **Simpan & Dapatkan Saran:** Simpan hasil ke history atau minta saran dari AI.

---

## Author
**Christopher Daniel Suryanaga**  
Capstone Project 4 – AI Engineer (Purwadhika)  
