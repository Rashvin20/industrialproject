import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from edsr import EDSR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EDSR(scale=2).to(device)
model.load_state_dict(torch.load("edsr_model.pth"))  # Load the saved model weights
model.eval()  # Set model to evaluation mode

# Preprocess the input image
def preprocess_image(image_path, hr_size=64):
    img = Image.open(image_path).convert('RGB')  # Open image and convert to RGB
    img = img.resize((hr_size, hr_size), Image.BICUBIC)  # Resize to match model input size
    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img).unsqueeze(0)  # Add batch dimension
    return img_tensor

# Perform super-resolution
def super_resolve(model, img_tensor, scale=2):
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        sr_img = model(img_tensor)
    return sr_img

# Postprocess the output image
def postprocess_image(sr_img):
    sr_img = sr_img.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
    return sr_img

# Input image path (replace with your own image path)
image_path = r"C:\Users\rashv\Downloads\test1.jpeg"
  # Replace with the path to your image

# Preprocess the image
img_tensor = preprocess_image(image_path, hr_size=64)

# Perform super-resolution
sr_img = super_resolve(model, img_tensor, scale=2)

# Postprocess and display the result
sr_img = postprocess_image(sr_img)

# Show original image and super-resolved image
original_img = Image.open(image_path).convert('RGB')
original_img = original_img.resize((sr_img.shape[1], sr_img.shape[0]), Image.BICUBIC)
original_img = np.array(original_img)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(original_img)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(sr_img)
plt.title("Super-Resolved Image")
plt.axis('off')

plt.show()
