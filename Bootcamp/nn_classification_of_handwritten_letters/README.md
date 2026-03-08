# 🧠 EMNISTN: High-Performance Handwritten Letter Classification

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Accuracy](https://img.shields.io/badge/Test_Accuracy-94.87%25-brightgreen)

<br>

## 📌 Project Overview
Developed for the **Codecademy ML Bootcamp Contest (Jan 2026)**, this project focused on high-efficiency Deep Learning. The challenge required building a neural network to classify handwritten letters (EMNIST) from scratch, strictly limited to <1M parameters.

My architecture, EMNISTNet, is a highly optimized CNN that achieves **94.87% test accuracy with only ~697k parameters**. Key technical implementations included Label Smoothing for better generalization, Dropout for regularization, and advanced feature extraction layers. Adhering to production standards, the model prioritizes a high accuracy-to-parameter ratio, making it ideal for edge computing or resource-constrained environments.

<br>

## 🚀 Key Results
| Metric | Performance | Notes |
| :--- | :--- | :--- |
| **Test Accuracy** | **94.87%** | High generalization (Validation Acc: 95.26%) |
| **Model Size** | **0.7 MB** | 697,594 Parameters (70% of budget) |
| **Inference** | **GPU/CPU** | Auto-detects hardware via `predict.py` |

<br>

## 🛠️ Technical Approach & MLOps
To achieve top-tier performance within constraints, I implemented:

* **Architecture:** Custom VGG-style CNN with Modular `ConvBlocks` (GELU Activations + BatchNorm).
* **Optimization:** `AdamW` optimizer + `OneCycleLR` scheduler for superconvergence (15 epochs).
* **Data Engineering:** Custom pipeline with `FixOrientation`, `RandomRotation`, and `GaussianNoise` for robustness.
* **Regularization:** Heavy Dropout + **Label Smoothing (0.1)** to handle ambiguous letters (e.g., 'I' vs 'l').
* **Reproducibility:** SHA256 integrity checks and automated model checkpointing.

<br>

## 📊 Visualizations
### Training Dynamics
![Training Curves](./images/Model_Training_Dynamics.png) 

### Confusion Matrix
![Confusion Matrix](./images/Confusion_Matrix.png)

<br>

## 💻 How to Run
1. Clone the repository.
   ```bash
   git clone https://github.com/MiltonValleL/CodeCademy-DataScience-ML-Portfolio.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run inference: To test the model on random samples and verify the architecture:
   ```bash
   python predict.py
   ```
*Note: The script automatically detects if a GPU (CUDA) is available for acceleration.*

<br>
<br>

## 🤝 Contact
I am a Data Science enthusiast passionate about Machine Learning and statistical modeling. 

If you have any questions about this project or would like to connect, feel free to reach out!

<br>

---
**Author:** Milton Rodolfo Valle Lora

**LinkedIn:** [Please click here](https://www.linkedin.com/in/miltonvallelora/)

---
