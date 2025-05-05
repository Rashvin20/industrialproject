# 🛰️ Satellite Image Super-Resolution with Deep Learning

> Enhancing low-resolution satellite imagery using state-of-the-art super-resolution models: **SRCNN**, **EDSR**, and **ESRGAN**.

---

## 📌 Overview

Satellite imagery plays a vital role in industries like **agriculture**, **urban planning**, and **real estate**. However, obtaining high-resolution (HR) satellite images is often costly, limited by licensing, and restricted in availability.

This project aims to **bridge the resolution gap** using deep learning-based **Super-Resolution (SR)** techniques — enabling accurate visual interpretation without relying on expensive HR datasets.

### 🚀 Collaborating Partner  
**Knight Frank**, a global real estate firm , is supporting this project. They rely on geospatial data for **property valuation**, **risk assessment**, and **site analysis**, all of which benefit greatly from enhanced satellite imagery.

---

## 🎯 Objectives

- ✅ Train and compare **SRCNN**, **EDSR**, and **ESRGAN** models for single-image super-resolution (SISR).
- ✅ Apply models to satellite-like image data and evaluate using **PSNR**, **SSIM**, and visual quality.
- ✅ Enable decision-making for large-scale real estate analysis using upscaled geospatial imagery.

---

## 🧠 Methods

| Model   | Description |
|---------|-------------|
| 🔹 **SRCNN** | First deep-learning SR model. Simple 3-layer CNN. |
| 🔸 **EDSR**  | Deep residual blocks without batch norm. Improved details. |
| 🔺 **ESRGAN** | GAN-based, produces sharper, more realistic textures. *(In Progress)* |

---

## 📂 Dataset

For prototyping, we use **CIFAR-10** resized to mimic satellite imagery. Future versions will incorporate real datasets like:

- UC Merced Land Use
- DeepGlobe
- SpaceNet

---

## 🛠️ Setup & Installation

```bash
git clone https://github.com/your-username/satellite-super-resolution.git
cd satellite-super-resolution
pip install -r requirements.txt
