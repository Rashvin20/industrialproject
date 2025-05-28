# 🛰️ Satellite Image Super-Resolution with Deep Learning

> Enhancing low-resolution satellite imagery using state-of-the-art super-resolution models: **SRCNN**, **EDSR**, and **ESRGAN**.

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
- ✅ Support decision-making in large-scale real estate and environmental analysis through improved geospatial image clarity.

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
