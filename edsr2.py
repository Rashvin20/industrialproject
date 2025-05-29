import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tacoreader
import rasterio as rio
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import os

# ---------- 1. Dataset ----------
class Sen2NaipDataset(Dataset):
    def __init__(self, split="tacofoundation:sen2naipv2-unet", indices=None, patch_size=64, scale=4, color=True):
        self.dataset = tacoreader.load(split)
        self.indices = indices if indices else range(len(self.dataset))
        self.patch_size = patch_size
        self.scale = scale
        self.color = color
        self.hr_patch_size = patch_size * scale

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx = self.indices[idx]
        lr_ref = self.dataset.read(sample_idx).read(0)
        hr_ref = self.dataset.read(sample_idx).read(1)

        with rio.open(lr_ref) as src_lr, rio.open(hr_ref) as src_hr:
            lr_data = src_lr.read(window=rio.windows.Window(0, 0, self.patch_size, self.patch_size))
            hr_data = src_hr.read(window=rio.windows.Window(0, 0, self.hr_patch_size, self.hr_patch_size))

        if self.color:
            return self.to_tensor_rgb(lr_data), self.to_tensor_rgb(hr_data)
        else:
            return self.to_tensor_gray(lr_data), self.to_tensor_gray(hr_data)

    def to_tensor_gray(self, data, norm=3000.0):
        rgb = data[:3] / norm
        gray = 0.2989 * rgb[0] + 0.5870 * rgb[1] + 0.1140 * rgb[2]
        return torch.tensor(gray, dtype=torch.float32).unsqueeze(0)  # [1, H, W]

    def to_tensor_rgb(self, data, norm=3000.0):
        rgb = data[:3] / norm
        return torch.tensor(rgb, dtype=torch.float32)  # [3, H, W]


# ---------- 2. Model (EDSR) ----------
class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class EDSR(nn.Module):
    def __init__(self, scale=4, num_res_blocks=8, in_channels=3):
        super(EDSR, self).__init__()
        self.scale = scale
        self.head = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(64) for _ in range(num_res_blocks)])
        self.tail = nn.Sequential(
            nn.Conv2d(64, in_channels * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


# ---------- 3. Training ----------
def compute_psnr(output, target, max_pixel=1.0):
    mse = torch.mean((output - target) ** 2)
    return 100 if mse.item() == 0 else 20 * math.log10(max_pixel / math.sqrt(mse.item()))

def train_model(model, dataloader, device, num_epochs=5, save_path="edsr_epoch"):
    print("Begin Training")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epoch_losses = []
    epoch_psnrs = []

    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch+1}/{num_epochs}...")
        model.train()
        running_loss = 0.0
        psnr_accum = 0.0

        for i, (lr, hr) in enumerate(dataloader):
            lr, hr = lr.to(device), hr.to(device)

            optimizer.zero_grad()
            output = model(lr)
            loss = criterion(output, hr)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            psnr_accum += compute_psnr(output, hr)

        avg_loss = running_loss / len(dataloader)
        avg_psnr = psnr_accum / len(dataloader)

        epoch_losses.append(avg_loss)
        epoch_psnrs.append(avg_psnr)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, PSNR: {avg_psnr:.2f} dB")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"{save_path}_{epoch+1}.pth")
            print(f"Model saved as: {save_path}_{epoch+1}.pth")

    # Plot loss and PSNR
    plt.figure()
    plt.plot(range(1, num_epochs+1), epoch_losses, label='Loss')
    plt.plot(range(1, num_epochs+1), epoch_psnrs, label='PSNR')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Training Loss and PSNR')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('edsr_training_metrics.png')
    plt.show()

    print("Training completed. Displaying sample output...")
    test_random_sample(model, dataloader.dataset, device)


# ---------- 4. Inference & Visualization ----------
def test_random_sample(model, dataset, device):
    model.eval()
    idx = random.randint(0, len(dataset) - 1)
    lr, hr = dataset[idx]

    lr = lr.unsqueeze(0).to(device)
    hr = hr.unsqueeze(0).to(device)

    with torch.no_grad():
        sr = model(lr)

    lr_np = lr.squeeze().cpu().numpy().transpose(1, 2, 0)
    hr_np = hr.squeeze().cpu().numpy().transpose(1, 2, 0)
    sr_np = sr.squeeze().cpu().numpy().transpose(1, 2, 0)

    plt.figure(figsize=(15, 5))
    titles = ['Low-Resolution (Input)', 'High-Resolution (Ground Truth)', 'EDSR Output']
    for i, img in enumerate([lr_np, hr_np, sr_np]):
        plt.subplot(1, 3, i + 1)
        plt.imshow(img)
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# ---------- 5. Entrypoint ----------
def main(train=True, model_path="edsr_epoch_10.pth"):
    print("Launching main training/testing procedure...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = EDSR(scale=4, in_channels=3).to(device)

    if train:
        print("Loading training dataset...")
        train_dataset = Sen2NaipDataset(indices=range(100), color=True)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        print("Starting model training...")
        train_model(model, train_loader, device)
    else:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")
        print("Loading trained model and testing dataset...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        test_dataset = Sen2NaipDataset(indices=range(10), color=True)
        print("Running inference on test sample...")
        test_random_sample(model, test_dataset, device)

# Run training or testing
main(train=True)
