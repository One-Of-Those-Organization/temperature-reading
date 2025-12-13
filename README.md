# Temperature Reading System - Internet of Things (IoT)

Sistem berbasis web untuk membaca dan mendigitalkan nilai temperatur pada panel **Refrigerated Container (Reefer Container)** secara otomatis menggunakan kamera dan kecerdasan buatan (*Deep Learning*).

Sistem ini menggantikan metode OCR konvensional (seperti Tesseract) dengan arsitektur **CRNN (Convolutional Recurrent Neural Network)** yang lebih tangguh terhadap gangguan visual (blur, pencahayaan minim, dll).

## Daftar Isi
- [Project Overview](#-project-overview)
- [Metodologi & Arsitektur AI](#-metodologi--arsitektur-ai)
- [Struktur Project](#-struktur-project)
- [Prasarat (Requirements)](#-prasarat-requirements)
- [Cara Menjalankan (Installation & Run)](#-cara-menjalankan-installation--run)
- [Cara Penggunaan](#-cara-penggunaan)
- [Tim Pengembang](#-tim-pengembang)

---

## Project Overview

Dalam industri logistik *cold chain*, pemantauan suhu kontainer sangat krusial. Proyek ini memungkinkan pengguna untuk mengunggah foto panel suhu, memilih area angka (Setpoint & Air Temperature) melalui antarmuka web, dan mendapatkan hasil pembacaan digital secara *real-time*.

**Fitur Utama:**
* **Single Image Upload:** Pengguna cukup mengunggah 1 foto panel utuh.
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
* **CTC Loss (Connectionist Temporal Classification):** Layer output yang memungkinkan model memprediksi teks tanpa perlu memisahkan (segmentasi) setiap karakter secara manual.

---

## Struktur Project

Struktur direktori proyek ini adalah sebagai berikut:

```text
temperature-reading/
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