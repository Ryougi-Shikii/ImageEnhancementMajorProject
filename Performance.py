from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

def calculate_metrics(enhanced_img, gt_path):
    gt_img = cv2.imread(gt_path)
    if gt_img is None:
        return None, None
    
    # Ensure they are the same size
    if enhanced_img.shape != gt_img.shape:
        gt_img = cv2.resize(gt_img, (enhanced_img.shape[1], enhanced_img.shape[0]))

    psnr_val = psnr_metric(gt_img, enhanced_img)
    ssim_val = ssim_metric(gt_img, enhanced_img, channel_axis=2)
    return psnr_val, ssim_val
    
import os
import cv2
import torch
import numpy as np
from RetinexFormer_arch import RetinexFormer

# ---------------- CONFIG ---------------- #
MODEL_PATH = "retinex_1.pth"         # Path to your trained weights
INPUT_DIR = "lol_dataset/eval15/low" # Use the 'low' folder from eval15
OUTPUT_DIR = "results_eval15"        # New folder for output
GT_DIR = "lol_dataset/eval15/high"   # Path to 'high' for metric comparison
STRIDE = 32  # overlap for smooth blending
PATCH_SIZE = 512 # 128 and 256
'''
->for 128
Average PSNR: 16.99
Average SSIM: 0.7154
-> for 257
Average PSNR: 18.44
Average SSIM: 0.7271

'''

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD MODEL ---------------- #
def load_model():
    model = RetinexFormer(n_feat=64, stage=2)  # match training config
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

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = load_model()
    
    psnr_list, ssim_list = [], []

    for img_name in os.listdir(INPUT_DIR):
        input_path = os.path.join(INPUT_DIR, img_name)
        gt_path = os.path.join(GT_DIR, img_name) # Assuming filenames match
        
        img = cv2.imread(input_path)
        if img is None: continue

        enhanced = enhance_image(model, img)
        cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), enhanced)
        
        # Calculate metrics
        p, s = calculate_metrics(enhanced, gt_path)
        if p is not None:
            psnr_list.append(p)
            ssim_list.append(s)
            print(f"{img_name} -> PSNR: {p:.2f}, SSIM: {s:.4f}")

    if psnr_list:
        print("\n--- Final Results ---")
        print(f"Average PSNR: {np.mean(psnr_list):.2f}")
        print(f"Average SSIM: {np.mean(ssim_list):.4f}")

if __name__ == "__main__":
    main()





import os
import cv2
import matplotlib.pyplot as plt

def create_comparison_grid(low_dir, enhanced_dir, high_dir, output_grid_path, num_images=15):
    """
    Creates a grid visualization: Low-light | Enhanced | Ground Truth
    """
    images = [f for f in os.listdir(enhanced_dir) if f.endswith(('.png', '.jpg', '.jpeg'))][:]
    
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)

    titles = ["Input (Low-light)", "Output (RetinexFormer)", "Ground Truth (High)"]

    for i, img_name in enumerate(images):
        # Paths
        paths = [
            os.path.join(low_dir, img_name),
            os.path.join(enhanced_dir, img_name),
            os.path.join(high_dir, img_name)
        ]

        for j, path in enumerate(paths):
            img = cv2.imread(path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[i, j].imshow(img)
            
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(titles[j], fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_grid_path, bbox_inches='tight', dpi=200)
    print(f"Grid saved to {output_grid_path}")
    plt.show()

# --- Run it ---
create_comparison_grid(
    low_dir="lol_dataset/eval15/low",
    enhanced_dir="results_eval15",
    high_dir="lol_dataset/eval15/high",
    output_grid_path="comparison_grid.png"
)