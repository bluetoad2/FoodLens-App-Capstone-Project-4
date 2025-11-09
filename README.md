# FoodLens: Aplikasi Analisis Nutrisi Berbasis AI

**FoodLens** adalah aplikasi web berbasis Streamlit yang menggunakan kecerdasan buatan (AI) untuk mendeteksi makanan dari gambar dan memberikan analisis nutrisi secara instan. Aplikasi ini bertujuan untuk membantu pengguna melacak asupan kalori dan makronutrien, memantau riwayat konsumsi, serta mendapatkan saran gizi yang dipersonalisasi.

---

## Ringkasan Proyek

Proyek ini merupakan implementasi dari sistem *Retrieval-Augmented Generation* (RAG) yang dikemas dalam aplikasi web interaktif. FoodLens memanfaatkan model deteksi objek **YOLO** untuk mengidentifikasi makanan, kemudian mengambil data nutrisi yang relevan, dan menyajikannya dalam antarmuka yang mudah digunakan.

Tujuan utamanya adalah menyediakan alat yang praktis bagi pengguna untuk memantau diet mereka dengan lebih mudah, hanya dengan menggunakan foto makanan.

Seluruh aplikasi dibangun menggunakan **Python** dengan framework **Streamlit**. Untuk fitur cerdasnya, aplikasi ini mengintegrasikan model **OpenAI (gpt-4o-mini)** melalui **LangChain** untuk memberikan saran gizi.

---

## Fitur Utama

- **Analisis Gambar Tunggal:**
  - Unggah satu gambar (`.jpg`, `.png`, `.jpeg`).
  - Menampilkan gambar asli dan gambar hasil deteksi dengan *bounding box* di sekitar makanan.
  - Menampilkan tabel rincian nutrisi (kalori, protein, karbohidrat, lemak) dan tingkat kepercayaan deteksi.
  - Menampilkan ringkasan total nutrisi dari semua makanan yang terdeteksi.

- **Asisten Gizi AI:**
  - Tombol "Dapatkan Saran AI & Resep" yang muncul setelah analisis gambar berhasil.
  - Menggunakan model `gpt-4o-mini` dari OpenAI untuk memberikan rekomendasi makanan alternatif yang lebih sehat dan ide resep praktis.
  - Menampilkan estimasi penggunaan token dan biaya untuk setiap permintaan ke API OpenAI.

- **Dashboard Riwayat Nutrisi:**
  - Halaman khusus untuk melihat riwayat semua makanan yang telah dianalisis dan disimpan.
  - Filter data berdasarkan rentang waktu (7 hari terakhir, 30 hari terakhir, atau semua riwayat).
  - Visualisasi data interaktif menggunakan **Plotly** menunjukkan:
    - Tren asupan kalori harian.
    - Komposisi makronutrien (dalam bentuk diagram lingkaran).
    - Frekuensi makanan yang paling sering dikonsumsi.
  - Tombol untuk menghapus seluruh riwayat jika pengguna ingin memulai dari awal.

- **Analisis Gambar Massal (Batch):**
  - Fitur untuk mengunggah beberapa file gambar sekaligus atau dalam bentuk file `.zip`.
  - Menampilkan progress bar selama proses analisis.
  - Hasil dari semua gambar digabungkan dalam satu tabel ringkasan.
  - Opsi untuk mengunduh laporan lengkap dalam format **CSV** atau **PDF**.

---

## Alur Kerja Aplikasi

1.  **Input Pengguna**: Pengguna mengunggah satu atau beberapa gambar makanan melalui antarmuka Streamlit.
2.  **Deteksi Objek**: Gambar diproses oleh model **YOLO** (`best.pt`) yang telah dilatih untuk mengenali berbagai jenis makanan.
3.  **Ekstraksi Informasi**: Nama makanan dan informasi nutrisi (kalori, protein, karbohidrat, lemak) diekstrak dari data yang telah ditentukan sebelumnya.
4.  **Perhitungan & Tampilan**: Aplikasi menghitung dan menampilkan rincian nutrisi per item dan ringkasan totalnya. Aplikasi juga menampilkan gambar asli dengan kotak pembatas yang ditumpangkan pada item yang terdeteksi.
5.  **Saran AI (Opsional)**: Jika pengguna meminta saran, daftar makanan yang terdeteksi akan dikirim ke **API OpenAI** melalui **LangChain** untuk menghasilkan saran diet dan resep yang dipersonalisasi.
6.  **Pencatatan Riwayat (Opsional)**: Jika pengguna memilih untuk menyimpan hasilnya, data akan ditulis ke dalam database **SQLite** lokal.

---

## Struktur Proyek

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
│   └── foodlens.db         # File database SQLite (dibuat otomatis)
│
├── models/
│   └── best.pt             # File model deteksi objek YOLO
│
├── FoodLens App.py         # Skrip utama aplikasi Streamlit
├── requirements.txt        # Daftar library yang dibutuhkan
├── conclusion.md           # Catatan kesimpulan proyek
└── README.md               # File ini
```

---

## Lingkungan & Kredensial

Untuk menggunakan fitur Asisten Gizi AI, Anda perlu membuat file `.streamlit/secrets.toml` dan memasukkan kunci API OpenAI Anda di dalamnya:

```toml
OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

---

## Dependensi

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

## Cara Menjalankan Aplikasi

1.  **Pastikan File Model Ada:**
    Letakkan file model `best.pt` di dalam folder `models/`.

2.  **Buat Virtual Environment (Direkomendasikan):**
    ```bash
    python -m venv venv
    ```
    Aktifkan environment:
    ```bash
    # Untuk Windows
    .\venv\Scripts\activate
    
    # Untuk macOS/Linux
    source venv/bin/activate
    ```

3.  **Instal Dependensi:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Jalankan Aplikasi:**
    Buka terminal, arahkan ke direktori utama proyek, dan jalankan perintah berikut:
    ```bash
    streamlit run "FoodLens App.py"
    ```
    Aplikasi akan terbuka secara otomatis di browser Anda.
