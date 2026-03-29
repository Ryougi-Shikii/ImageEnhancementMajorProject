import os
import cv2
import torch
import numpy as np
from RetinexFormer_arch import RetinexFormer

# ---------------- CONFIG ---------------- #
MODEL_PATH = "retinex.pth"      # path to your trained model
INPUT_DIR = "input_images"      # folder with input images
OUTPUT_DIR = "results"          # folder to save outputs

PATCH_SIZE = 128
STRIDE = 64   # overlap (important)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- LOAD MODEL ---------------- #
def load_model():
    model = RetinexFormer(n_feat=16, stage=1)  # match your training config
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# ---------------- PATCH INFERENCE ---------------- #
def enhance_image(model, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    img = img.astype(np.float32) / 255.0

    output = np.zeros((h, w, 3), dtype=np.float32)
    weight = np.zeros((h, w, 3), dtype=np.float32)

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

            # remove padding
            out = out[:ph, :pw]

            # add to output with weight
            output[i:i+ph, j:j+pw] += out
            weight[i:i+ph, j:j+pw] += 1.0

    # normalize overlapping areas
    output = output / weight
    output = np.clip(output, 0, 1)

    output = (output * 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    return output


# ---------------- MAIN ---------------- #
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = load_model()

    for img_name in os.listdir(INPUT_DIR):
        input_path = os.path.join(INPUT_DIR, img_name)
        output_path = os.path.join(OUTPUT_DIR, img_name)

        img = cv2.imread(input_path)

        if img is None:
            print(f"Skipping {img_name}")
            continue

        enhanced = enhance_image(model, img)

        cv2.imwrite(output_path, enhanced)
        print(f"Processed: {img_name}")

    print("Done. Check results folder.")


if __name__ == "__main__":
    main()