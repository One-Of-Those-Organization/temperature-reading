import os
import io
import json
import base64
import traceback
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
from flask import Flask, render_template, request, jsonify
from datetime import datetime
import paho.mqtt.client as mqtt

# ==========================================
# MQTT Stuff
# ==========================================

MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC = "ukdc/iot/temparature01"
MQTT_KEEPALIVE = 60

def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print(f"Connected to MQTT Broker:  {MQTT_BROKER}")
        client.subscribe(MQTT_TOPIC)
    else:
        print(f"Failed to connect, return code {reason_code}")

def on_message(client, userdata, msg):
    """Called when a message is received from subscribed topic"""
    # print(f"Received message on {msg.topic}: {msg.payload.decode()}")

def on_publish(client, userdata, mid, reason_code, properties):
    """Called when message is published"""
    print(f"Message published (mid:  {mid})")

def on_disconnect(client, userdata, disconnect_flags, reason_code, properties):
    """Called when client disconnects"""
    print(f"Disconnected from MQTT broker (code: {reason_code})")

# Initialize MQTT Client
mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqttc.on_connect = on_connect
mqttc.on_message = on_message
mqttc.on_publish = on_publish
mqttc.on_disconnect = on_disconnect

# Connect to MQTT Broker
try:
    print(f"Connecting to MQTT Broker:  {MQTT_BROKER}:{MQTT_PORT}")
    mqttc.connect(MQTT_BROKER, MQTT_PORT, MQTT_KEEPALIVE)
    mqttc.loop_start()
    print("MQTT loop started")
except Exception as e:
    print(f"MQTT Connection Error: {e}")

def publish_to_mqtt(data):
    """Publish data to MQTT broker"""
    try:
        payload = json.dumps(data)
        result = mqttc.publish(MQTT_TOPIC, payload, qos=1)
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"Data published to MQTT topic '{MQTT_TOPIC}'")
            return True
        else:
            print(f"Failed to publish to MQTT:  {result.rc}")
            return False
    except Exception as e:
        print(f"MQTT Publish Error: {e}")
        return False

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

def normalize_region(r, img_w, img_h):
    """
    Normalize cropper.js region data
    """
    x = int(round(r.get("x", 0)))
    y = int(round(r.get("y", 0)))
    w = int(round(r.get("width", 0)))
    h = int(round(r.get("height", 0)))

    # clamp supaya tidak keluar image
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))

    return x, y, w, h

LAST_IMAGE_B64 = None
LAST_REGIONS = []   # [{x,y,w,h}, {x,y,w,h}]
STATE_FILE = "data/state.json"

if os.path.exists(STATE_FILE):
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            LAST_IMAGE_B64 = state.get("image")
            LAST_REGIONS = state.get("regions", [])
            print("State loaded")
    except:
        print("State rusak, abaikan")

def save_state():
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump({
            "image": LAST_IMAGE_B64,
            "regions": LAST_REGIONS
        }, f, indent=4)

app = Flask(__name__)
data = "data/data.json"

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

@app.route('/new')
def newroute():
    return render_template('new.html')

@app.route('/api/state', methods=['GET'])
def get_state():
    if LAST_IMAGE_B64 is None:
        return jsonify({
            "status": 1,
            "message": "Belum ada image"
        })

    return jsonify({
        "status": 0,
        "image": LAST_IMAGE_B64,
        "regions": LAST_REGIONS
    })

@app.route('/api/set-image', methods=['POST'])
def set_image():
    if model is None:
        return jsonify({
            "status": 1,
            "message": "Model belum dimuat"
            }), 500

    try:
        payload = request.get_json()

        image_b64 = payload.get('image', '')
        regions = payload.get('region', [])
        source_type = payload.get('source', 'camera')

        # Validasi
        if not image_b64 or not regions:
            return jsonify({
                "status": 2,
                "message": "Image atau region kosong"
                }), 400

        if len(regions) != 2:
            return jsonify({
                "status": 3,
                "message": "Untuk saat ini region harus 2 rect"
                }), 406

        # Decode base64
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]

        image_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        results = []

        global LAST_IMAGE_B64, LAST_REGIONS
        normalized_regions = []
        img_w, img_h = img.size

        for r in regions:
            x, y, w, h = normalize_region(r, img_w, img_h)
            normalized_regions.append({
                "x": x,
                "y": y,
                "w": w,
                "h": h
            })

        LAST_IMAGE_B64 = image_b64
        LAST_REGIONS = normalized_regions
        save_state()

        img_w, img_h = img.size

        for idx, r in enumerate(regions):
            x, y, w, h = normalize_region(r, img_w, img_h)
            crop_img = img.crop((x, y, x + w, y + h))

            # Convert crop to bytes
            buf = io.BytesIO()
            crop_img.save(buf, format="PNG")
            crop_bytes = buf.getvalue()

            text = predict_image(crop_bytes)

            results.append({
                "index": idx,
                "value": text
                })

        # Susun data hasil
        result_data = {
                "setpoint_code": results[0]["value"],
                "air_temperature": results[1]["value"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": source_type
                }

        mqtt_success = publish_to_mqtt(result_data)
        result_data["mqtt_published"] = mqtt_success

        return jsonify({
            "status": 0,
            "message": "Image processed",
            "data": result_data
            })

    except Exception as e:
        print("Set Image Error:", e)
        print(traceback.format_exc())
        return jsonify({
            "status": 9,
            "message": str(e)
            }), 500


@app.route('/api/read-meter', methods=['POST'])
def read_meter():
    # Cek Model
    if model is None:
        return jsonify({
            "status": 1,
            "message": "Model belum dimuat"
        }), 500

    try:
        data = request.get_json()

        # Ambil Data
        setpoint_b64 = data.get('setpoint_image', '')
        air_temp_b64 = data.get('air_temp_image', '')
        source_type = data.get('source', 'camera')

        if not setpoint_b64 or not air_temp_b64:
            return jsonify({
                "status": 2,
                "message": "Gambar kurang!"
            }), 400

        # Decode Base64
        if "," in setpoint_b64: setpoint_b64 = setpoint_b64.split(",")[1]
        if "," in air_temp_b64: air_temp_b64 = air_temp_b64.split(",")[1]

        sp_bytes = base64.b64decode(setpoint_b64)
        air_bytes = base64.b64decode(air_temp_b64)

        # Prediksi CRNN
        setpoint_result = predict_image(sp_bytes)
        air_temp_result = predict_image(air_bytes)

        # Buat Object Data Baru
        result_data = {
            "setpoint_code": setpoint_result,
            "air_temperature": air_temp_result,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": source_type,
        }

        mqtt_success = publish_to_mqtt(result_data)

        # Write to data/data.json
        data_pointer = "data/data.json"
        history_data = []

        # Check Existed File
        if os.path.exists(data_pointer):
            try:
                with open(data_pointer, "r") as f:
                    file_content = json.load(f)

                    # Pastikan formatnya List
                    if isinstance(file_content, list):
                        history_data = file_content
                    else:
                        history_data = [file_content]
            except Exception as e:
                print(f"Warning: File lama rusak/kosong, membuat baru. Error: {e}")
                history_data = []

        # Append New Data
        history_data.insert(0, result_data)

        # Set Limit Data
        if len(history_data) >= 100:
            history_data = history_data[:100]

        # Write a new Data
        try:
            os.makedirs(os.path.dirname(data_pointer), exist_ok=True)  # Create New if NOT Exist
            with open(data_pointer, "w") as f:
                json.dump(history_data, f, indent=4)
            print(f"Data tersimpan: {result_data['timestamp']}")
        except Exception as e:
            print(f"Gagal tulis data: {e}")

        result_data['mqtt_published'] = mqtt_success

        return jsonify({
            "status": 0,
            "message": "Send Data",
            "data": result_data
        })

    except Exception as e:
        print(f"Server Error: {e}")
        print(traceback.format_exc())
        return jsonify({"status": 4, "message": str(e)}), 500

@app.route('/api/get-meter', methods=['GET'])
def get_meter():
    data_path = "data/data.json"

    # Check if data is existed
    if not os.path.exists(data_path):
        return jsonify({
            "status": 5,
            "message": "Data tidak ditemukan pada Server !"
        })

    # If data existed
    try:
        with open(data_path, "r") as f:
            data = json.load(f)

        return jsonify({
            "status": 0,
            "message": "Send Data",
            "data": data
        })
    except Exception as e:
        return jsonify({
            "status": 6,
            "message": f"Server Error: {str(e)}",
            "data": "Tanya Fufufafa"
        })

if __name__ == '__main__':
    # Jalankan server di port 5000 atau ganti port lainnya
    print("Server berjalan di http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
