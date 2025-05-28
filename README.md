# 🛰️ Satellite Image Super-Resolution with Deep Learning

> Enhancing low-resolution satellite imagery using  super-resolution models: **SRCNN**, **EDSR**, and **ESRGAN**.

---

## 📌 Overview

Satellite imagery plays a vital role in industries like **agriculture**, **urban planning**, and **real estate**. However, obtaining high-resolution (HR) satellite images is often costly, limited by licensing, and restricted in availability.

This project aims to **bridge the resolution gap** using deep learning-based **Super-Resolution (SR)** techniques — enabling accurate visual interpretation without relying on expensive HR datasets.

### 🚀 Collaborating Partner  
**Knight Frank**, a global real estate firm, is supporting this project. They rely on geospatial data for **property valuation**, **risk assessment**, and **site analysis**, all of which benefit greatly from enhanced satellite imagery.

---

## 🎯 Objectives

- ✅ Train and compare **SRCNN**, **EDSR**, and **ESRGAN** models for single-image super-resolution (SISR).
- ✅ Apply models to real satellite image data and evaluate performance using **PSNR**, **SSIM**, and **perceptual loss**.
- ✅ **Develop an interactive web interface** that allows users to upload low-resolution imagery and download super-resolved outputs in **GeoTIFF** format.

---

## 🧠 Methods

| Model   | Description |
|---------|-------------|
| 🔹 **SRCNN** | First deep-learning SR model. Simple 3-layer CNN. |
| 🔸 **EDSR**  | Deep residual blocks without batch norm. Great for structural detail. |
| 🔺 **ESRGAN** | GAN-based, produces sharper, more realistic textures. |

---

## 📂 Dataset

We use the **Sen2NAIPv2** dataset hosted on Hugging Face:

- 🛰️ **Sentinel-2** (Low-Resolution, multi-spectral satellite imagery)  
- 🗺️ **NAIP** (High-Resolution aerial imagery from USDA)

These are paired and aligned to train and evaluate the super-resolution models.

---

## 🛠️ Setup & Installation

```bash
git clone https://github.com/Rashvin20/industrialproject.git
cd industrialproject
pip install -r requirements.txt
```

---

## ▶️ Running the Models

Each model is located in the `sr_techniques/` folder. You can run them individually as follows:

```bash
cd sr_techniques

# Run SRCNN
python srcnn_model.py

# Run EDSR
python edsr_model.py

# Run ESRGAN
python esrgan_model.py
```

Model weights and training graphs (PSNR, SSIM, perceptual loss) are also saved in this folder.

---

## 🌐 Launching the Interface

The Streamlit-based web interface is located in the `frontend/` folder. Launch it with:

```bash
cd frontend
streamlit run app.py
```

The interface allows users to:
- Upload low-resolution satellite images
- Apply any of the trained models
- Download results in **GeoTIFF** format for GIS compatibility

---

Meng Robotics and Ai  
University College London (UCL)  

