# Temperature Reading System - Internet of Things (IoT)

Sistem berbasis web untuk membaca dan mendigitalkan nilai temperatur pada panel **Refrigerated Container (Reefer Container)** secara otomatis menggunakan kamera dan kecerdasan buatan (*Deep Learning*).

Sistem ini menggantikan metode OCR konvensional (seperti Tesseract) dengan arsitektur **CRNN (Convolutional Recurrent Neural Network)** yang lebih tangguh terhadap gangguan visual (blur, pencahayaan minim, dll).

---

## Daftar Isi
- [Project Overview](#project-overview)
- [Metodologi & Arsitektur AI](#metodologi--arsitektur-ai)
- [Struktur Project](#struktur-project)
- [Prasarat (Requirements)](#prasarat-requirements)
- [Cara Menjalankan (Installation & Run)](#cara-menjalankan-installation--run)
- [Cara Penggunaan](#cara-penggunaan)
- [API](#api)
- [Diketahui/Masalah](#diketahuimasalah)
- [License](#license)

---

## Project Overview

Dalam industri logistik *cold chain*, pemantauan suhu kontainer sangat krusial. Proyek ini memungkinkan pengguna untuk mengunggah foto panel suhu, memilih area angka (Setpoint & Air Temperature) melalui antarmuka web, dan mendapatkan hasil pembacaan digital secara *real-time*.

**Fitur Utama**:
- **Single Image Upload:** Pengguna cukup mengunggah 1 foto panel utuh.
- **Live Camera:** Pengguna dapat menggunakan kamera secara langsung dan otomatis mengambil foto setiap 5 menit.
- **Smart Cropping:** Antarmuka web menggunakan `Cropper.js` untuk memotong area spesifik secara interaktif (_crop manual_).
- **Deep Learning Inference:** Backend menggunakan model PyTorch (`.pt`) yang dilatih khusus untuk membaca *7-segment display* pada panel LCD/LED.
- **High Accuracy:** Model mampu mengenali karakter angka `0-9`, simbol minus `-`, dan titik `.` untuk membaca suhu negatif dan desimal.

---

## Metodologi & Arsitektur AI

Model cerdas yang digunakan menggunakan **PyTorch** dengan tahapan:
1. **Preprocessing:**
    - Grayscale, autocontrast, resize height 64px, pad width 192px, normalization.
2. **Model:**
    - **CRNN:** 3 blok CNN, Bi-GRU, CTC Loss untuk prediksi urutan karakter tanpa segmentasi presisi.
3. **Output:**
    - Hasil prediksi dikirim lewat API, disimpan sebagai riwayat (JSON) oleh backend.

---

## Struktur Project

```text
temperature-reading/
│
│── data/
│   └── data.json            # Seluruh history (100 data terbaru)
│
├── model/
│   └── lcd_best.pt          # Bobot model PyTorch (CRNN)
│
├── static/
│   └── ukdc.png             # Aset (logo, dsb)
│
├── templates/
│   └── index.html           # Frontend web (HTML + JS + Tailwind CSS)
│
├── main.py                  # Backend utama (Flask)
├── README.md                # Ini file dokumentasi
└── requirements.txt         # Library Python
```

---

## Prasarat (Requirements)
- Python 3.10+
- pip, virtualenv
- Camera device (untuk fitur Live Camera)
- Sistem operasi: Linux/macOS/Windows

---

## Cara Menjalankan (Installation & Run)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```
Pastikan model berada di `model/lcd_best.pt`.

---

## Cara Penggunaan

### Upload Gambar
1. Buka UI web (templates/index.html).
2. Upload foto panel LCD/LED atau pakai kamera.
3. Crop manual **Setpoint** dan **Air Temperature** menggunakan Cropper.js.
4. Klik "Proses Data" dan hasil ditampilkan serta disimpan ke `data/data.json`.

### Live Camera
1. Aktifkan kamera dan berikan izin browser.
2. Crop manual area angka.
3. Tekan mulai, sistem akan capture foto tiap ±5 menit dan proses otomatis.

---

## API

- `POST /api/read-meter`  
  Kirim gambar/crop untuk inferensi model, hasil disimpan di data/data.json.

- `GET /api/get-meter`  
  Ambil riwayat (up to 100 data terbaru) untuk keperluan tampilan/integrasi.

- `POST /api/config`  
  Simpan region/crop config dari frontend.

- `GET /api/get-coords`  
  Mendapatkan konfigurasi crop terakhir/set dari backend.

- `POST /api/process`  
  Proses utama: kirim gambar yang sudah di-crop manual untuk inferensi.

---

## Diketahui/Masalah

- **Isu: Restore otomatis crop area dari koordinat (coords) dan gambar base64 ke Cropper.js**  
  TIDAK berhasil pada sebagian kasus (khusus di project ini).  
  User _harus_ crop manual (drag) setiap kali gambar di-load,  
  karena cropping otomatis memakai coords JSON/backend **tidak stabil/gagal** di frontend (biarpun sudah sesuai teori best-practice, kemungkinan bug/limitation Cropper.js versi tertentu).

- **No professional todo-list**  
  Tidak ada implementasi todo/kanban/fitur manajemen profesional di repo ini.