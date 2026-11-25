# Data-Efficient Image Classification: Evaluating ResNet-18, ViT-Tiny, and MobileViT-XXS under Low Data Regimes

**Authors:** Rahul Desai (Veermata Jijabai Technological Institute, Mumbai) and Eric Sakk (Morgan State University, Baltimore)  
**Venue:** Accepted at Computer Vision Conference (CVC) 2027  
**Colab Notebook:** [Open in Google Colab](https://colab.research.google.com/drive/1uxiEgHmpA0s9jOwp0mdKM8Pq2kA4Zu2d?usp=sharing)

---

## Abstract

Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) represent two distinct architectural paradigms in computer vision. While ViTs utilize multi-head self-attention to capture long-range contextual dependencies, their lack of structural inductive biases reduces sample efficiency when trained on small datasets. Conversely, CNNs incorporate translation equivariance and localized receptive fields, enabling data-efficient feature extraction. This paper presents a comparative empirical evaluation of three lightweight models: ResNet-18 (pure CNN), ViT-Tiny (pure transformer), and MobileViT-XXS (CNN-transformer hybrid). All architectures were trained from scratch on CIFAR-10 subsets consisting of 1%, 10%, 25%, 50%, 75%, and 100% training fractions across multiple independent trials under a fixed 20-epoch budget. We evaluate convergence rates, top-1 test accuracy, class-wise performance, parameter efficiency, and GPU execution time. ResNet-18 consistently achieved higher accuracy than ViT-Tiny and MobileViT-XXS across all data regimes, maintaining a 5.9 to 12.9 percentage point lead over ViT-Tiny and a 13.5 to 29.2 percentage point lead over MobileViT-XXS. In addition, ResNet-18 demonstrated faster convergence and lower total training latency than ViT-Tiny on larger regimes despite having roughly twice the parameter count. Under the ultra-low 1% regime (50 images/class), ResNet-18 achieved 30.9% test accuracy, outperforming ViT-Tiny (25.0%) and MobileViT-XXS (17.4%). Furthermore, MobileViT-XXS retained 32.1% of its full-data accuracy at 1% data, compared to 33.8% for ViT-Tiny and 37.1% for ResNet-18. These findings show that a lower parameter count does not necessarily confer sample efficiency, underscoring the practical utility of CNN inductive biases for data-constrained vision tasks.

---

## Project Summary

### Overview
This repository contains the complete experimental pipeline used to benchmark sample efficiency, training dynamics, and computational latency across three computer vision architectures trained from scratch without transfer learning:
* **Pure CNN:** ResNet-18 (11.17M parameters)
* **Pure Vision Transformer:** ViT-Tiny with patch size P = 4 (5.36M parameters)
* **Hybrid CNN-Transformer:** MobileViT-XXS (0.95M parameters)

### Experimental Setup
* **Dataset:** CIFAR-10 (32 × 32 RGB images across 10 classes), stratified into balanced training fractions of **1%** (500 samples), **10%** (5k), **25%** (12.5k), **50%** (25k), **75%** (37.5k), and **100%** (50k)
* **Training Protocol:** Random initialization (from scratch), 20 epochs, AdamW optimizer (learning rate = 1.0 × 10⁻⁴, weight decay = 1.0 × 10⁻⁴), batch size 64, and cross-entropy loss
* **Reproducibility:** Evaluated across 3 independent random seeds (S ∈ {42, 100, 999}) on an NVIDIA Tesla T4 GPU (16 GB VRAM) using PyTorch 2.2.0 and `timm`

### Key Results (Mean ± SD Top-1 Test Accuracy)

| Training Data Fraction | ResNet-18 (11.2M) | ViT-Tiny (5.4M) | MobileViT-XXS (0.95M) |
| :--- | :---: | :---: | :---: |
| **1% (500 samples)** | **30.9 ± 1.2%** | 25.0 ± 1.4% | 17.4 ± 1.1% |
| **10% (5k samples)** | **57.2 ± 1.1%** | 44.3 ± 1.7% | 34.1 ± 1.4% |
| **25% (12.5k samples)** | **65.4 ± 0.9%** | 57.1 ± 1.2% | 42.2 ± 1.1% |
| **50% (25k samples)** | **76.1 ± 0.7%** | 64.3 ± 0.8% | 49.0 ± 0.9% |
| **75% (37.5k samples)** | **80.2 ± 0.6%** | 69.2 ± 0.7% | 52.3 ± 0.8% |
| **100% (50k samples)** | **83.4 ± 0.5%** | 74.1 ± 0.6% | 54.2 ± 0.7% |

### Core Takeaways
1. **Inductive Biases Drive Sample Efficiency:** ResNet-18 outperformed both transformer-based models across every data regime and retained **37.1%** of its full-data accuracy when trained on just 1% of the dataset (compared to 33.8% for ViT-Tiny and 32.1% for MobileViT-XXS).
2. **Parameter Count ≠ Faster Training:** Despite having more than twice the parameters of ViT-Tiny, ResNet-18 trained up to **32.2% faster** on larger data splits (900s vs. 1,190s at 100% data) due to mature cuDNN convolution kernel fusion versus memory-bound self-attention operations.
3. **Resolution Stem Bottlenecks:** MobileViT-XXS achieved the lowest latency (58.3s–510s) and smallest footprint, but its default stride-32 downsampling (designed for 224 × 224 inputs) collapsed 32 × 32 images into 1 × 1 spatial maps before pooling, limiting classification accuracy.
