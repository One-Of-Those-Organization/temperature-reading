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
# MQTT CONFIG
# ==========================================
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC = "ukdc/iot/temparature01"
MQTT_KEEPALIVE = 60

# ==========================================
# GLOBAL STATE
# ==========================================
LAST_IMAGE_B64 = None
LAST_REGIONS = []   # [{x,y,w,h}, {x,y,w,h}]
STATE_FILE = "data/state.json"
DATA_FILE = "data/data.json"

# Load previous state if exists
if os.path.exists(STATE_FILE):
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            LAST_IMAGE_B64 = state.get("image")
            CROP_SETPOINT_B64 = state.get("crop_setpoint")
            CROP_AIRTEMP_B64 = state.get("crop_airtemp")
            LAST_REGIONS = state.get("regions", [])
        print("State lengkap berhasil diload")
    except Exception as e:
        print("State rusak, abaikan:", e)

def save_state():
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    state_data = {
        "image": LAST_IMAGE_B64,
        "crop_setpoint": CROP_SETPOINT_B64,
        "crop_airtemp": CROP_AIRTEMP_B64,
        "regions": LAST_REGIONS
    }
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state_data, f, indent=4)
        print("State tersimpan dengan lengkap")
    except Exception as e:
        print("Gagal menyimpan state:", e)

# ==========================================
# MQTT CALLBACKS
# ==========================================
def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print(f"Connected to MQTT Broker: {MQTT_BROKER}")
        client.subscribe(MQTT_TOPIC)
    else:
        print(f"Failed to connect, return code {reason_code}")

def on_message(client, userdata, msg):
    pass  # optional debug

def on_publish(client, userdata, mid, reason_code, properties):
    print(f"Message published (mid: {mid})")

def on_disconnect(client, userdata, disconnect_flags, reason_code, properties):
    print(f"Disconnected from MQTT broker (code: {reason_code})")

# Initialize MQTT Client
mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqttc.on_connect = on_connect
mqttc.on_message = on_message
mqttc.on_publish = on_publish
mqttc.on_disconnect = on_disconnect

try:
    mqttc.connect(MQTT_BROKER, MQTT_PORT, MQTT_KEEPALIVE)
    mqttc.loop_start()
    print("MQTT loop started")
except Exception as e:
    print(f"MQTT Connection Error: {e}")

def publish_to_mqtt(data):
    try:
        payload = json.dumps(data)
        result = mqttc.publish(MQTT_TOPIC, payload, qos=1)
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"Data published to MQTT topic '{MQTT_TOPIC}'")
            return True
        else:
            print(f"Failed to publish to MQTT: {result.rc}")
            return False
    except Exception as e:
        print(f"MQTT Publish Error: {e}")
        return False

# ==========================================
# IMAGE & CRNN SETUP
# ==========================================
ALPHABET = "0123456789-."
BLANK = "<BLANK>"
itos = {i: ch for i, ch in enumerate(ALPHABET)}
itos[len(itos)] = BLANK
NUM_CLASSES = len(itos)

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
        feats = feats.permute(0, 3, 1, 2).reshape(B, W, C * H)
        feats = feats.permute(1, 0, 2)
        out, _ = self.rnn(feats)
        logits = self.head(out)
        return logits

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

def pad_to_width(img, target_width=192):
    w, h = img.size
    if w >= target_width:
        return img.resize((target_width, h))
    pad_left = (target_width - w) // 2
    pad_right = target_width - w - pad_left
    return ImageOps.expand(img, border=(pad_left, 0, pad_right, 0), fill=128)

transform_pipeline = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Lambda(ImageOps.autocontrast),
    transforms.Resize(64),
    transforms.Lambda(lambda img: pad_to_width(img, target_width=192)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "model/lcd_best.pt"
model = None
try:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = CRNN(NUM_CLASSES).to(DEVICE)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("Model berhasil dimuat!")
except Exception as e:
    print(f"ERROR MEMUAT MODEL: {e}")

def predict_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = transform_pipeline(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(img_tensor)
        return ctc_decode(logits)[0]
    except Exception as e:
        print(f"Prediction Error: {e}")
        return "Error"

def normalize_region(r, img_w, img_h):
    x = int(round(r.get("x", 0)))
    y = int(round(r.get("y", 0)))
    w = int(round(r.get("width", r.get("w", 0))))
    h = int(round(r.get("height", r.get("h", 0))))
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return x, y, w, h

# ==========================================
# FLASK APP
# ==========================================
app = Flask(__name__)

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# New route for configuration
@app.route('/new')
def newroute():
    return render_template('new.html')

# API to get current state
@app.route('/api/state', methods=['GET'])
def get_state():
    if LAST_IMAGE_B64 is None:
        return jsonify({"status": 1,
                        "message": "Belum ada image"
                        }), 404

    return jsonify({"status": 0,
                    "image": LAST_IMAGE_B64,
                    "crop_setpoint": CROP_SETPOINT_B64,
                    "crop_airtemp": CROP_AIRTEMP_B64,
                    "regions": LAST_REGIONS
                    })

# API to set configuration (crop regions)
@app.route('/api/config', methods=['POST'])
def set_config():
    try:
        payload = request.get_json()
        image_b64 = payload.get('image')
        regions = payload.get('region')

        if not image_b64 or not regions or len(regions) != 2:
            return jsonify({"status": 2,
                            "message": "Invalid config"
                            }), 400

        if "," in image_b64: image_b64 = image_b64.split(",")[1]
        img_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_w, img_h = img.size
        normalized = []
        for r in regions:
            x, y, w, h = normalize_region(r, img_w, img_h)
            normalized.append({"x": x, "y": y, "w": w, "h": h})
        global LAST_IMAGE_B64, LAST_REGIONS, CROP_SETPOINT_B64, CROP_AIRTEMP_B64
        LAST_IMAGE_B64 = image_b64
        LAST_REGIONS = normalized
        CROP_SETPOINT_B64 = payload.get("setpoint_crop_image", CROP_SETPOINT_B64)
        CROP_AIRTEMP_B64 = payload.get("airtemp_crop_image", CROP_AIRTEMP_B64)
        save_state()
        return jsonify({"status": 0,
                        "message": "Configuration saved",
                        "regions": normalized
                        })
    except Exception as e:
        return jsonify({"status": 3,
                        "message": f"Server Error: {str(e)}"
                        }), 500

# API to Process image and return predictions
@app.route('/api/process', methods=['POST'])
def process_image():
    if model is None:
        return jsonify({"status": 4,
                        "message": "Model not loaded"
                        }), 500
    if not LAST_REGIONS:
        return jsonify({"status": 5,
                        "message": "No configuration set"
                        }), 400
    try:
        payload = request.get_json()
        image_b64 = payload.get('image')
        source = payload.get('source', 'unknown')
        if not image_b64:
            return jsonify({"status": 6,
                            "message": "Image missing"
                            }), 400
        if "," in image_b64: image_b64 = image_b64.split(",")[1]
        img_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_w, img_h = img.size
        results = []
        for r in LAST_REGIONS:
            x, y, w, h = normalize_region(r, img_w, img_h)
            crop = img.crop((x, y, x + w, y + h))
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            text = predict_image(buf.getvalue())
            results.append(text)
        result_data = {
            "setpoint_code": results[0],
            "air_temperature": results[1],
            "source": source,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        mqtt_success = publish_to_mqtt(result_data)
        result_data["mqtt_published"] = mqtt_success
        # Write history to data.json
        history_data = []
        if os.path.exists(DATA_FILE):
            try:
                with open(DATA_FILE, "r", encoding="utf-8") as f:
                    file_content = json.load(f)
                    history_data = file_content if isinstance(file_content, list) else [file_content]
            except:
                history_data = []
        history_data.insert(0, result_data)
        history_data = history_data[:100]
        try:
            os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
            with open(DATA_FILE, "w", encoding="utf-8") as f:
                json.dump(history_data, f, ensure_ascii=False, indent=4)
        except Exception as e_write:
            print("Failed to write data.json:", e_write)
        return jsonify({"status": 0, "data": result_data})
    except Exception as e:
        return jsonify({"status": 7,
                        "message": str(e)
                        }), 500

# API to get crop coordinates
@app.route('/api/get-coords', methods=['GET'])
def get_coords():
    if not os.path.exists(STATE_FILE):
        return jsonify({"status": 8,
                        "message": "No crop configuration found",
                        "data": None
                        }), 404
    try:
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
        if "regions" not in data:
            return jsonify({"status": 9,
                            "message": "Invalid config file",
                            "data": None
                            }), 400
        return jsonify({"status": 0, "message": "Crop config loaded", "data": data["regions"]})
    except Exception as e:
        return jsonify({"status": 10,
                        "message": f"Server error: {str(e)}",
                        "data": None
                        }), 500

# Public API to get meter data history
@app.route('/api/get-meter', methods=['GET'])
def get_meter():
    if not os.path.exists(DATA_FILE):
        return jsonify({"status": 11,
                        "message": "Data tidak ditemukan pada Server!"
                        }), 400
    try:
        with open(DATA_FILE, "r") as f:
            data_content = json.load(f)
        return jsonify({"status": 0, "message": "Send Data", "data": data_content})
    except Exception as e:
        return jsonify({"status": 12,
                        "message": f"Server Error: {str(e)}",
                        "data": None
                        }), 500

# Server start on localhost:5000
if __name__ == '__main__':
    print("Server berjalan di http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)