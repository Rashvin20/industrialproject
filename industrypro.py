import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

# Define the EDSR Model
class EDSR(nn.Module):
    def __init__(self, scale=2):
        super(EDSR, self).__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 3 * (scale ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        return x

# Dataset class to handle CIFAR10
class CIFAR10_SR(Dataset):
    def __init__(self, train=True, scale=2, hr_size=64):
        self.dataset = datasets.CIFAR10(root='./data', train=train, download=True)
        self.hr_size = hr_size  # High-resolution target size
        self.scale = scale
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        # Resize to create high-resolution (HR) target
        hr = TF.resize(img, [self.hr_size, self.hr_size], interpolation=InterpolationMode.BICUBIC)
        hr = self.to_tensor(hr)

        # Downscale to create low-resolution (LR) input
        lr = TF.resize(hr, [self.hr_size // self.scale, self.hr_size // self.scale], interpolation=InterpolationMode.BICUBIC)
        lr = TF.resize(lr, [self.hr_size, self.hr_size], interpolation=InterpolationMode.BICUBIC)

        return lr, hr

    def __len__(self):
        return len(self.dataset)

# Training function
# Training function
import torch.nn.functional as F

def train_model(model, dataloader, num_epochs=10, lr=1e-4):
    criterion = nn.L1Loss()  # L1 loss for super-resolution
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Check for GPU availability
    model.to(device)  # Move the model to the available device
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for lr_imgs, hr_imgs in dataloader:
            # Move images to the device (GPU or CPU)
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            # Resize the high-resolution images to match the output size of the model
            hr_imgs = F.interpolate(hr_imgs, size=(lr_imgs.size(2) * scale, lr_imgs.size(3) * scale), mode='bicubic', align_corners=False)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            sr_imgs = model(lr_imgs)

            # Calculate loss
            loss = criterion(sr_imgs, hr_imgs)
            loss.backward()

            # Update weights
            optimizer.step()

            # Track the running loss
            running_loss += loss.item()

        # Print statistics
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}')


# Set up data loader
scale = 2
train_dataset = CIFAR10_SR(train=True, scale=scale, hr_size=64)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Initialize and train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EDSR(scale=scale).to(device)
train_model(model, train_loader, num_epochs=10)

# Save the trained model
torch.save(model.state_dict(), "edsr_model.pth")
