import os
import io
import base64
import traceback
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
from flask import Flask, render_template, request, jsonify
from datetime import datetime

# ==========================================
# CRNN Setup
# ==========================================

# Get Numeric Only
ALPHABET = "0123456789-."
BLANK = "<BLANK>"
itos = {i: ch for i, ch in enumerate(ALPHABET)}
itos[len(itos)] = BLANK
NUM_CLASSES = len(itos)

# CRNN Module
class CRNN(nn.Module):
    def __init__(self, num_classes, in_ch=1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 192, 3, padding=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True),
        )
        self.rnn = nn.GRU(
            input_size=192 * 16,
            hidden_size=192,
            num_layers=2,
            bidirectional=True,
            dropout=0.1,
            batch_first=False
        )
        self.head = nn.Sequential(
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, num_classes)
        )

    def forward(self, x):
        feats = self.cnn(x)
        B, C, H, W = feats.shape
        # Permute agar sesuai dimensi RNN (Sequence, Batch, Features)
        feats = feats.permute(0, 3, 1, 2).reshape(B, W, C * H)
        feats = feats.permute(1, 0, 2)
        out, _ = self.rnn(feats)
        logits = self.head(out)
        return logits

# Translate model to text
def ctc_decode(logits):
    probs = logits.softmax(-1)
    best = probs.argmax(-1)
    results = []
    for b in range(best.shape[1]):
        seq = best[:, b].tolist()
        dedup = []
        prev = None
        for s in seq:
            if s == NUM_CLASSES - 1:
                prev = s
                continue
            if s != prev:
                dedup.append(s)
            prev = s
        text = "".join(itos[i] for i in dedup)
        results.append(text)
    return results

# Adapt Image Width Padding
def pad_to_width(img, target_width=192):
    w, h = img.size
    if w >= target_width:
        return img.resize((target_width, h))
    pad_left = (target_width - w) // 2
    pad_right = target_width - w - pad_left
    return ImageOps.expand(img, border=(pad_left, 0, pad_right, 0), fill=128)

# Pipeline Preprocessing (Grayscale -> Resize -> Pad -> Normalize)
transform_pipeline = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Lambda(ImageOps.autocontrast),
    transforms.Resize(64),
    transforms.Lambda(lambda img: pad_to_width(img, target_width=192)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# ==========================================
# FLASK APP SETUP
# ==========================================
app = Flask(__name__)

# Konfigurasi Device & Model Path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # EARLY WARNING : CHANGE this to DEVICE = "cpu" if you have some trouble with cuda
MODEL_PATH = "model/lcd_best.pt"
model = None # Fallback if Model path not found

# Load Model saat aplikasi dijalankan
try:
    print(f"Memuat model dari {MODEL_PATH} ke {DEVICE}...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    # Inisialisasi model
    model = CRNN(NUM_CLASSES).to(DEVICE)

    # Load state dict (bobot)
    # Menangani kemungkinan key 'model' atau langsung state_dict
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()  # Set ke mode evaluasi (bukan training)
    print("Model berhasil dimuat!")
except Exception as e:
    print(f"ERROR MEMUAT MODEL: {e}")
    print("Pastikan file 'lcd_best.pt' ada di folder yang sama.")

def predict_image(image_bytes):
    """Fungsi helper untuk memprediksi satu gambar bytes"""
    try:
        # Buka gambar dari bytes
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Transformasi gambar ke Tensor
        img_tensor = transform_pipeline(img).unsqueeze(0).to(DEVICE)

        # Inference
        with torch.no_grad():
            logits = model(img_tensor)
            result_text = ctc_decode(logits)[0]

        return result_text
    except Exception as e:
        print(f"Prediction Error: {e}")
        return "Error"

# ==========================================
# 3. ROUTE API & HALAMAN
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/read-meter', methods=['POST'])
def read_meter():
    """
    Endpoint utama untuk menerima 2 gambar hasil crop.
    Menerima JSON:
    {
        "setpoint_image": "data:image/jpeg;base64,.....",
        "air_temp_image": "data:image/jpeg;base64,....."
    }
    """
    # Cek apakah model sudah siap
    if model is None:
        return jsonify({"status": 0, "message": "Model belum dimuat (file model.pt tidak ditemukan?)"}), 500

    try:
        data = request.get_json()

        # 1. Ambil string base64
        setpoint_b64 = data.get('setpoint_image', '')
        air_temp_b64 = data.get('air_temp_image', '')

        # If the Payload is NULL then
        if not setpoint_b64 or not air_temp_b64:
            return jsonify({"status": 0, "message": "Kedua gambar (Setpoint & Air Temp) harus dikirim!"}), 400

        # 2. Bersihkan header base64 (data:image/jpeg;base64,...)
        if "," in setpoint_b64: setpoint_b64 = setpoint_b64.split(",")[1]
        if "," in air_temp_b64: air_temp_b64 = air_temp_b64.split(",")[1]

        # 3. Decode ke Bytes
        setpoint_bytes = base64.b64decode(setpoint_b64)
        air_temp_bytes = base64.b64decode(air_temp_b64)

        # 4. Prediksi dengan Model AI
        # Model dipanggil dua kali untuk masing-masing potongan gambar
        print("Menganalisis Setpoint...")
        setpoint_result = predict_image(setpoint_bytes)

        print("Menganalisis Air Temp...")
        air_temp_result = predict_image(air_temp_bytes)

        # 5. Format Timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"Hasil -> SP: {setpoint_result}, Air: {air_temp_result}")

        # 6. Kirim Respon JSON
        response = {
            "status": 1,
            "message": "Sukses membaca meter",
            "data": {
                "setpoint_code": setpoint_result,
                "air_temperature": air_temp_result,
                "timestamp": current_time
            }
        }
        return jsonify(response)

    except Exception as e:
        print("Server Error:")
        print(traceback.format_exc())
        return jsonify({"status": 0, "message": f"Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    # Jalankan server di port 5000 atau ganti port lainnya
    print("Server berjalan di http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)