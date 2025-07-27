# Adversarial_Attack
Multi-Class Detection of Adversarial Perturbation with Local Histogram Equalization and High-Pass Filters


---

# 🔐 Adversarial Attack Type Classifier using GhostNet and CIFAR-100

This project builds a robust image classifier that identifies whether an image is clean or has been modified by an adversarial attack. It leverages "GhostNet" as the backbone model and operates on the "CIFAR-100" dataset with preprocessing via "Local Histogram Equalization (LHE)" and "High-pass filtering". Adversarial examples are generated using popular attacks like FGSM, PGD, BIM, AutoAttack, and random noise.

---

## 📌 Features

* ✅ Train on clean + adversarially perturbed images.
* ✅ Detect attack type (multi-class classification).
* ✅ Apply image preprocessing (LHE + High-pass filter).
* ✅ Use pretrained "GhostNet" model from `timm`.
* ✅ Support for mixed precision training via `torch.cuda.amp`.
* ✅ Visualize confusion matrix and correlation matrix.
* ✅ Evaluate classifier on unseen data.

---

## 📁 Dataset

* CIFAR-100: Automatically downloaded via `torchvision.datasets`.
* Splits:

  * 80% of CIFAR-100 train set for training.
  * Remaining 20% used for validation.
  * Test set used to generate adversarial examples.

---

## 🧪 Adversarial Attacks Used

* [x] FGSM (Fast Gradient Sign Method)
* [x] PGD (Projected Gradient Descent)
* [x] BIM (Basic Iterative Method)
* [x] AutoAttack
* [x] Random Gaussian Noise

All adversarial examples are labeled according to the type of attack.

---

## 🧠 Model

* Architecture: `ghostnet_100` from [`timm`](https://github.com/rwightman/pytorch-image-models)
* Output classes: 6 (Clean + 5 attack types)
* Loss: Cross-entropy with label smoothing
* Optimizer: Adam
* LR Scheduler: StepLR
* Mixed-precision training via `GradScaler` and `autocast`

---

## 🔄 Preprocessing Pipeline

1. **Local Histogram Equalization** (per channel)
2. **High-pass sharpening filter**

This helps amplify subtle perturbations caused by adversarial attacks.

---

## 📊 Evaluation

* Final test accuracy reported.
* Classification report (precision, recall, f1-score).
* Confusion matrix & correlation heatmap for attack types.

---

## 🖼 Sample Output

<img width="921" height="516" alt="image" src="https://github.com/user-attachments/assets/135ed30f-268f-4e8e-a8d7-7e597cf685db" />


Confusion Matrix:
<img width="921" height="548" alt="image" src="https://github.com/user-attachments/assets/14ac2039-23e9-4a46-bd3b-7d93cbc06c89" />


Correlation Matrix:
<img width="921" height="535" alt="image" src="https://github.com/user-attachments/assets/d41d877f-0bce-4258-ab2c-496adcdcf8cf" />

Classification Report:
<img width="849" height="538" alt="image" src="https://github.com/user-attachments/assets/bad80f94-b687-4c8a-a748-fe68617569b7" />


---

## 🚀 How to Run

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/adv-attack-detector.git
cd adv-attack-detector
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

#### Required packages

```text
torch
torchvision
timm
opencv-python
numpy
pandas
matplotlib
seaborn
scikit-learn
tqdm
torchattacks
```

### 3. Run the training script

```bash
python main.py
```

> This will automatically download CIFAR-100, generate adversarial examples, preprocess data, and train the classifier.

---

## 📎 Results Summary

![Screenshot 2025-07-03 124329](https://github.com/user-attachments/assets/58aabbdf-5f6f-4ca5-b88c-63abf8fb87a8)
![Screenshot 2025-07-03 124248](https://github.com/user-attachments/assets/97356ad4-e017-4217-b71c-62dc3cc9aaa3)
![WhatsApp Image 2025-07-03 at 12 43 08_aaa60671](https://github.com/user-attachments/assets/820328bf-1f31-42fe-b8ce-a2a93c3232fc)
![WhatsApp Image 2025-07-03 at 12 43 18_fadc8144](https://github.com/user-attachments/assets/d1aa1a67-d97e-40a5-bd33-c724f8d2648f)





## 📌 Future Improvements

* Add support for more attack types (e.g., CW, DeepFool).
* Augment training with more preprocessing variants.
* Try other lightweight models like EfficientNet or MobileNetV3.
* Deploy with TorchScript or ONNX for real-time use.

---

## 📜 License

This project is licensed under the Shibam License.





