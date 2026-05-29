

# install imp libraries 
# !pip install pyngrok flask_cors
# save auth token 
# !ngrok config add-authtoken <your ngrok token>

# cell 1 
import os
import io
import cv2
import base64
import threading
import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok

# Import your model architecture
from RetinexFormer_arch import RetinexFormer

# ---- Configuration ----
MODEL_PATH = "retinex_1.pth"
PATCH_SIZE = 256
STRIDE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Model Initialization ----
model = RetinexFormer(n_feat=64, stage=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Thread lock to prevent concurrent CUDA inference conflicts
model_lock = threading.Lock()

def enhance_image(model, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    img = img.astype(np.float32) / 255.0

    output = np.zeros((h, w, 3), dtype=np.float32)
    weight = np.zeros((h, w, 3), dtype=np.float32)

    # 1. Create a smooth 2D Hann window weight matrix
    patch_weight_1d = np.hanning(PATCH_SIZE)
    patch_weight = np.outer(patch_weight_1d, patch_weight_1d)
    patch_weight = np.expand_dims(patch_weight, axis=2)  # Shape: (128, 128, 1)

    for i in range(0, h, STRIDE):
        for j in range(0, w, STRIDE):
            # Formulate patch slices safely
            patch = img[i : i + PATCH_SIZE, j : j + PATCH_SIZE]
            ph, pw = patch.shape[:2]

            # Pad patch if it's on the edge to maintain PATCH_SIZE
            if ph < PATCH_SIZE or pw < PATCH_SIZE:
                pad = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.float32)
                pad[:ph, :pw] = patch
                patch = pad

            # Prepare tensor for model
            patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

            # Thread-safe inference
            with model_lock:
                with torch.no_grad():
                    out = model(patch_tensor)

            # Convert back to numpy
            out = out.squeeze().permute(1, 2, 0).cpu().numpy()
            out = np.clip(out, 0, 1)

            # 2. FIX: Apply weights to full patch before cropping out padding
            weighted_out = out * patch_weight
            
            # Accumulate full or cropped segments back into the final canvas
            output[i : i + ph, j : j + pw] += weighted_out[:ph, :pw]
            weight[i : i + ph, j : j + pw] += patch_weight[:ph, :pw]

    # 3. FIX: Add epsilon (1e-8) to prevent NaN errors from division by zero
    output = output / (weight + 1e-8)
    output = np.clip(output, 0, 1)
    
    # Convert back to standard image format
    output = (output * 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    return output


# cell 2
# ---- Flask API Server ----
app = Flask(__name__)
CORS(app)

@app.route("/enhance", methods=["POST"])
def enhance():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "Missing 'image' field"}), 400

    try:
        # Decode base64 string directly into image
        img_bytes = base64.b64decode(data["image"])
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Could not decode image"}), 400

        # Run enhancement pipeline
        enhanced = enhance_image(model, img)

        # Encode back to base64
        _, buffer = cv2.imencode(".jpg", enhanced)
        result_b64 = base64.b64encode(buffer).decode("utf-8")
        
        return jsonify({"enhanced_image": result_b64})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # If running locally or in environments like Colab, uncomment below to expose via ngrok:
    public_url = ngrok.connect(5000).public_url
    print(" * ngrok tunnel available at:", public_url)
    
    app.run(host="0.0.0.0", port=5000)



'''
usual output: 

 * ngrok tunnel available at: https://freemason-astonish-angelic.ngrok-free.dev
 * Serving Flask app '__main__'
 * Debug mode: off
INFO:werkzeug:WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://172.28.0.12:5000
INFO:werkzeug:Press CTRL+C to quit
INFO:werkzeug:127.0.0.1 - - [29/May/2026 12:42:28] "OPTIONS /enhance HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [29/May/2026 12:46:05] "POST /enhance HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [29/May/2026 12:48:39] "OPTIONS /enhance HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [29/May/2026 12:52:15] "POST /enhance HTTP/1.1" 200 -
'''



