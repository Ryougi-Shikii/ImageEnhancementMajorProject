# model enhance_retinex_1 -> n_feat=64, stage=2
import os
import cv2
import torch
import numpy as np

MODEL_PATH = "retinex_1.pth"   

PATCH_SIZE = 128
STRIDE = 32 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = RetinexFormer(n_feat=64, stage=2) 
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def enhance_image(model, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    img = img.astype(np.float32) / 255.0

    output = np.zeros((h, w, 3), dtype=np.float32)
    weight = np.zeros((h, w, 3), dtype=np.float32)

    # create smooth patch weight (Hann window)
    patch_weight_1d = np.hanning(PATCH_SIZE)
    patch_weight = np.outer(patch_weight_1d, patch_weight_1d)
    patch_weight = np.expand_dims(patch_weight, axis=2)  # shape (PATCH_SIZE, PATCH_SIZE, 1)

    for i in range(0, h, STRIDE):
        for j in range(0, w, STRIDE):
            patch = img[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
            ph, pw = patch.shape[:2]

            # pad if needed
            if ph < PATCH_SIZE or pw < PATCH_SIZE:
                pad = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.float32)
                pad[:ph, :pw] = patch
                patch = pad

            # to tensor
            patch_tensor = torch.from_numpy(patch).permute(2,0,1).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                out = model(patch_tensor)

            out = out.squeeze().permute(1,2,0).cpu().numpy()
            out = np.clip(out, 0, 1)
            out = out[:ph, :pw]

            # apply smooth weight
            w_patch = patch_weight[:ph, :pw]
            output[i:i+ph, j:j+pw] += out * w_patch
            weight[i:i+ph, j:j+pw] += w_patch

    # normalize overlapping areas
    output = output / weight
    output = np.clip(output, 0, 1)
    output = (output * 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    return output

# Enhance Image api point ( cell 2 )


from pyngrok import ngrok
import threading, base64, io
import numpy as np
import cv2
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from RetinexFormer_arch import RetinexFormer

# ---- Flask app ----
app = Flask(__name__)
CORS(app)

@app.route("/enhance", methods=["POST"])
def enhance():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "Missing 'image' field"}), 400

    try:
        img_bytes = base64.b64decode(data["image"])
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Could not decode image"}), 400

        enhanced = enhance_image(model, img)

        _, buffer = cv2.imencode(".jpg", enhanced)
        result_b64 = base64.b64encode(buffer).decode("utf-8")
        return jsonify({"enhanced_image": result_b64})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---- Start server + ngrok ----
public_url = ngrok.connect(5000)
print("Paste this URL into the frontend:")
print(str(public_url) + "/enhance") # paste with .../enhance at end

threading.Thread(target=lambda: app.run(port=5000)).start()


# add ngrok auth token
# !ngrok config add-authtoken token_here

