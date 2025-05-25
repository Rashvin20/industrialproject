import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np
from srcnn_model import SRCNN
from io import BytesIO
import os
import rasterio
from rasterio.transform import from_origin

# Set working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---------- GeoTIFF Writer ----------
def tensor_to_geotiff(tensor: torch.Tensor, filename: str = "enhanced_photo.tif"):
    sr_np = tensor.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    sr_np = np.clip(sr_np, 0, 1)
    sr_uint8 = (sr_np * 255).astype(np.uint8)  # 👈 make it viewable

    height, width, channels = sr_uint8.shape
    transform = from_origin(0, 0, 1, 1)
    crs = "EPSG:4326"

    with rasterio.open(
        filename,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=channels,
        dtype=sr_uint8.dtype,
        transform=transform,
        crs=crs
    ) as dst:
        for i in range(channels):
            dst.write(sr_uint8[:, :, i], i + 1)

    return filename


# ---------- Streamlit Setup ----------
st.set_page_config(page_title="AI Satellite Image Enhancer", layout="centered")
st.title("✨ AI Satellite Image Enhancer")
st.write("Upscale and enhance blurry satellite images using a AI techniques.")

# ---------- Load Model ----------
@st.cache_resource
def load_model(path="srcnn_epoch.pth"):
    model = SRCNN(in_channels=3, out_channels=3)  # ✅ color model
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ---------- Preprocessing ----------
def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    np_img = np.array(img).astype(np.float32)

    # Crop to multiple of 4 for bicubic upsampling alignment
    h, w, _ = np_img.shape
    h = (h // 4) * 4
    w = (w // 4) * 4
    np_img = np_img[:h, :w, :]

    # Normalize
    if np_img.max() <= 255:
        np_img /= 255.0
    else:
        np_img /= 3000.0

    tensor = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    return tensor

# ---------- Postprocessing ----------
def postprocess_tensor(tensor: torch.Tensor):
    array = tensor.squeeze(0).detach().cpu().numpy()
    array = np.clip(array.transpose(1, 2, 0), 0, 1)
    return Image.fromarray((array * 255).astype(np.uint8))

# ---------- File Upload ----------
uploaded_file = st.file_uploader("📤 Upload a GeoTIFF Image (RGB only)", type=["tiff"])

if uploaded_file:
    input_image = Image.open(uploaded_file).convert("RGB")
    input_tensor = preprocess_image(input_image)
    model = load_model()

    # Bicubic Upsample to match training input
    input_upsampled = torch.nn.functional.interpolate(
        input_tensor, scale_factor=4, mode="bicubic", align_corners=False
    )

    with st.spinner("🔧 Enhancing with SRCNN..."):
        with torch.no_grad():
            output_tensor = model(input_upsampled)
        sr_image = postprocess_tensor(output_tensor)

    # ---------- Comparison ----------
    st.markdown("### 🔍 Comparison")
    col1, col2 = st.columns(2)
    col1.image(input_image, caption="Original", use_container_width=True)
    col2.image(sr_image, caption="Enhanced", use_container_width=True)

    # ---------- PNG Download ----------
    buf = BytesIO()
    sr_image.save(buf, format="PNG")
    st.download_button(
        label="📥 Download Enhanced Image (PNG)",
        data=buf.getvalue(),
        file_name="enhanced_srcnn.png",
        mime="image/png"
    )

    # ---------- GeoTIFF Download ----------
    geo_path = "enhanced_srcnn.tif"
    tensor_to_geotiff(output_tensor, filename=geo_path)
    with open(geo_path, "rb") as f:
        st.download_button(
            label="📥 Download as GeoTIFF",
            data=f.read(),
            file_name="enhanced_srcnn.tiff",
            mime="image/tiff"
        )
