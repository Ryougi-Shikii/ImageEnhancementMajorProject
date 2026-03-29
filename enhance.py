import torch
import cv2
import os
from retinexformer_arch import RetinexFormer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model
model = RetinexFormer(n_feat=16, stage=1)
model.load_state_dict(torch.load("retinex.pth", map_location=device))
model.to(device)
model.eval()

input_dir = "input_images"
output_dir = "results"

os.makedirs(output_dir, exist_ok=True)

for img_name in os.listdir(input_dir):
    path = os.path.join(input_dir, img_name)

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128,128))

    tensor = torch.from_numpy(img).float().permute(2,0,1)/255.0
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)

    out = out.squeeze().permute(1,2,0).cpu().numpy()
    out = (out * 255).clip(0,255).astype("uint8")

    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(output_dir, img_name), out)

print("Done. Check results/")