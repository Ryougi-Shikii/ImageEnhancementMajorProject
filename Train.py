

# ------------- Defining Class to lear dataset ----------------------
import os, cv2, torch
from torch.utils.data import Dataset

class LOLDataset(Dataset):
    def __init__(self, low_dir, high_dir, size=256):
        self.low_paths = sorted([os.path.join(low_dir, f) for f in os.listdir(low_dir)])
        self.high_paths = sorted([os.path.join(high_dir, f) for f in os.listdir(high_dir)])
        self.size = size

    def __len__(self):
        return len(self.low_paths)

    def __getitem__(self, idx):
        low = cv2.imread(self.low_paths[idx])
        high = cv2.imread(self.high_paths[idx])

        low = cv2.cvtColor(low, cv2.COLOR_BGR2RGB)
        high = cv2.cvtColor(high, cv2.COLOR_BGR2RGB)

        low = cv2.resize(low, (256,256))
        high = cv2.resize(high, (256,256))

        low = torch.from_numpy(low).float().permute(2,0,1)/255.0
        high = torch.from_numpy(high).float().permute(2,0,1)/255.0

        return low, high


# ------------- Loading Retinex Arcitecture ----------------------
import torch
from RetinexFormer_arch import RetinexFormer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RetinexFormer(
    n_feat=16,      # ↓ from 32
    stage=1         # ↓ from 3 (VERY important)
).to(device)

# ------------- Loading dataset ----------------------
low_path = "/content/lol_dataset/our485/low"
high_path = "/content/lol_dataset/our485/high"


from torch.utils.data import DataLoader
dataset = LOLDataset(low_path, high_path)
loader = DataLoader(dataset, batch_size=1, shuffle=True)


# ------------- Training ----------------------
criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

'''
low = cv2.resize(low, (128,128))
high = cv2.resize(high, (128,128))
loader = DataLoader(dataset, batch_size=1, shuffle=True)
model = RetinexFormer(
    n_feat=16,      # ↓ from 32
    stage=1         # ↓ from 3 (VERY important)
).to(device)
'''

# ------------- Training loop ----------------------

epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for low, high in loader:
        low = low.to(device)
        high = high.to(device)

        output = model(low)

        loss = torch.nn.functional.l1_loss(output, high)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader)}")

    torch.save(model.state_dict(), "/content/retinex.pth")
