# Comparative Analysis of Face Recognition Models

A research-focused study evaluating the performance of **FaceNet**, **VGG-Face**, and **ArcFace** architectures on the LFW dataset.

## 🛡️ Key Features
- **Multi-Model Support**: Comparative benchmarking using `DeepFace`.
- **Privacy-Preserving**: AES-256 encryption for biometric template storage.
- **Scientific Metrics**: Automated calculation of **FAR**, **FRR**, and **EER (Equal Error Rate)**.
- **Visual Analytics**: Automatic generation of **ROC Curves** for performance visualization.

## 📈 Benchmarking Results
Based on a subset of the LFW dataset (25 identities):

| Model | Accuracy | EER |
| :--- | :--- | :--- |
| **FaceNet** | 93.8% | **0.076** |
| **ArcFace** | **94.4%** | 0.095 |
| **VGG-Face** | 91.0% | 0.105 |

## 🛠️ Usage
1. Initialize the dataset using a symbolic link to LFW.
2. Run `python evaluation.py` to generate the threshold sweep and ROC curves.