# Temperature Reading System - Internet of Things (IoT)

Sistem berbasis web untuk membaca dan mendigitalkan nilai temperatur pada panel **Refrigerated Container (Reefer Container)** secara otomatis menggunakan kamera dan kecerdasan buatan (*Deep Learning*).

Sistem ini menggantikan metode OCR konvensional (seperti Tesseract) dengan arsitektur **CRNN (Convolutional Recurrent Neural Network)** yang lebih tangguh terhadap gangguan visual (blur, pencahayaan minim, dll).

## Daftar Isi
- [Project Overview](##-project-overview)
- [Metodologi & Arsitektur AI](##-metodologi--arsitektur-ai)
- [Struktur Project](##-struktur-project)
- [Prasarat (Requirements)](##-prasarat-requirements)
- [Cara Menjalankan (Installation & Run)](##-cara-menjalankan-installation--run)
- [Cara Penggunaan](##-cara-penggunaan)

---

## Project Overview

Dalam industri logistik *cold chain*, pemantauan suhu kontainer sangat krusial. Proyek ini memungkinkan pengguna untuk mengunggah foto panel suhu, memilih area angka (Setpoint & Air Temperature) melalui antarmuka web, dan mendapatkan hasil pembacaan digital secara *real-time*.

**Fitur Utama:**
* **Single Image Upload:** Pengguna cukup mengunggah 1 foto panel utuh.
* **Live Camera:** Pengguna dapat menggunakan kamera secara langsung dan otomatis mengambil foto setiap 5 menit.
* **Smart Cropping:** Antarmuka web (Frontend) menggunakan `Cropper.js` untuk memotong area spesifik secara interaktif.
* **Deep Learning Inference:** Backend menggunakan model PyTorch (`.pt`) yang dilatih khusus untuk membaca *7-segment display* pada panel LCD/LED.
* **High Accuracy:** Model mampu mengenali karakter angka `0-9`, simbol minus `-`, dan titik `.` untuk membaca suhu negatif dan desimal.

---

## Metodologi & Arsitektur AI

Model cerdas yang digunakan dalam sistem ini dilatih menggunakan framework **PyTorch** dengan tahapan pemrosesan sebagai berikut:

### 1. Preprocessing Citra
Sebelum masuk ke model, potongan gambar (crop) diproses agar seragam:
* **Grayscale:** Konversi citra ke 1 channel warna (hitam putih).
* **Autocontrast:** Meningkatkan kontras untuk memperjelas bentuk angka.
* **Fixed Resize:** Tinggi gambar diubah menjadi **64 piksel**.
* **Padding:** Lebar gambar di-pad menjadi **192 piksel** (dengan warna abu-abu/hitam) untuk menjaga rasio aspek angka agar tidak gepeng.
* **Normalization:** Menormalisasi nilai piksel (mean 0.5, std 0.5) agar pembacaan model lebih stabil.

### 2. Arsitektur Model (CRNN)
Sistem menggunakan arsitektur *hybrid* yang menggabungkan CNN dan RNN:
* **CNN (Convolutional Neural Network):** Terdiri dari 3 blok konvolusi untuk mengekstrak fitur visual dari gambar (garis, lekukan).
* **RNN (Bi-directional GRU):** Membaca urutan fitur dari kiri-ke-kanan dan kanan-ke-kiri untuk memahami konteks urutan karakter.
* **CTC Loss (Connectionist Temporal Classification):** Layer output yang memungkinkan model memprediksi teks tanpa perlu memisahkan (segmentasi) 

### 3. Output / Result
Setelah menganalisis dari hasil CRNN, data tersebut akan dikirimkan ke api/read-meter:
* **Write Data:** Untuk dibaca dari hasil output menulis hasil ke data/data.json untuk menjadi sebuah laporan history
* **Unique Data:** Dari data JSON tersebut akan memiliki pembeda antara upload file dengan live camera dengan pengambilan 100 data terbaru.
* **Read Data**: Dari History data/data.json juga dapat diambil melalui api/get-meter untuk di implementasikan lebih bagus tampilannya.

---

## Struktur Project

Struktur direktori proyek ini adalah sebagai berikut:

```text
temperature-reading/
│
│── data/
│   └── data.json            # Keseluruhan History dari 100 data terbaru  
│
├── model/
│   └── lcd_best.pt          # File bobot model (Weight) hasil training PyTorch
│
├── static/
│   └── ukdc.png             # Aset gambar (Logo Universitas/Project)
│
├── templates/
│   └── index.html           # Frontend Interface (HTML + JS + Tailwind CSS)
│
├── main.py                  # Backend Server utama (Flask App)
├── README.md                # Dokumentasi Proyek ini
└── requirements.txt         # Daftar library Python yang dibutuhkan
```

---

## Prasarat (Requirements)
- Python 3.10+ (sesuaikan dengan versi di `requirements.txt`)
- Pip / venv untuk manajemen paket
- Camera device (opsional, untuk mode Live Camera)
- Akses jaringan untuk memuat aset/model jika perlu
- Sistem operasi: Linux / macOS / Windows (disarankan Linux/macOS untuk dev)

---

## Cara Menjalankan (Installation & Run)

### 1) Siapkan lingkungan virtual & dependency
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Jalankan aplikasi
```bash
python main.py
```
Secara default akan membuka server Flask (cek log di terminal untuk host/port). Jika ada variabel environment khusus (mis. `HOST`, `PORT`), sesuaikan sebelum menjalankan.

### 3) Struktur data & model
- Pastikan `model/lcd_best.pt` tersedia (bobot CRNN).
- File hasil pembacaan disimpan di `data/data.json` (otomatis dibuat/diperbarui).

---

## Cara Penggunaan

### Upload Gambar Tunggal
1. Buka halaman web (template `templates/index.html`).
2. Unggah foto panel LCD/LED.
3. Gunakan Cropper di UI untuk memilih area Setpoint & Air Temperature.
4. Submit dan hasil pembacaan akan ditampilkan serta data disimpan ke `data/data.json`.

### Live Camera
1. Aktifkan kamera pada browser dengan memberi ijin.
2. Lakukan crop manual pada area panel (Total 2 Panel).
3. Klik tombol mulai capture gambar.
4. Sistem akan mengambil foto berkala (±5 menit).
5. Hasil akan diproses dengan model CRNN dan dikirim ke `data/data.json`.

### List API
- `POST /api/read-meter` — kirim gambar (atau crop) untuk inferensi, hasil disimpan ke `data/data.json`.
- `GET /api/get-meter` — ambil riwayat (maks 100 data terbaru) untuk ditampilkan/diintegrasikan.