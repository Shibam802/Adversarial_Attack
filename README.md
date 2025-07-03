# Adversarial_Attack
Multi-Class Detection of Adversarial Perturbation with Local Histogram Equalization and High-Pass Filters
Here's a complete `README.md` file you can use for your GitHub repository. It clearly describes your project, dependencies, usage, and key results. You can copy and paste this into a `README.md` file in your repository:

---

# 🔐 Adversarial Attack Type Classifier using GhostNet and CIFAR-100

This project builds a robust image classifier that identifies whether an image is clean or has been modified by an adversarial attack. It leverages **GhostNet** as the backbone model and operates on the **CIFAR-100** dataset with preprocessing via **Local Histogram Equalization (LHE)** and **High-pass filtering**. Adversarial examples are generated using popular attacks like FGSM, PGD, BIM, AutoAttack, and random noise.

---

## 📌 Features

* ✅ Train on clean + adversarially perturbed images.
* ✅ Detect attack type (multi-class classification).
* ✅ Apply image preprocessing (LHE + High-pass filter).
* ✅ Use pretrained **GhostNet** model from `timm`.
* ✅ Support for mixed precision training via `torch.cuda.amp`.
* ✅ Visualize confusion matrix and correlation matrix.
* ✅ Evaluate classifier on unseen data.

---

## 📁 Dataset

* **CIFAR-100**: Automatically downloaded via `torchvision.datasets`.
* Splits:

  * **80%** of CIFAR-100 train set for training.
  * Remaining 20% used for validation.
  * **Test set** used to generate adversarial examples.

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

```text
Sample 1: True = FGSM, Predicted = PGD  
Sample 2: True = Clean, Predicted = Clean  
...
```

Confusion Matrix:
![Confusion Matrix](./images/confusion_matrix.png)

Correlation Matrix:
![Correlation Matrix](./images/correlation_matrix.png)

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

| Attack Type | Precision | Recall | F1-score |
| ----------- | --------- | ------ | -------- |
| Clean       | xx%       | xx%    | xx%      |
| FGSM        | xx%       | xx%    | xx%      |
| PGD         | xx%       | xx%    | xx%      |
| BIM         | xx%       | xx%    | xx%      |
| AutoAttack  | xx%       | xx%    | xx%      |
| RandomNoise | xx%       | xx%    | xx%      |

> *(Fill in with actual results after training)*

---

## 📌 Future Improvements

* Add support for more attack types (e.g., CW, DeepFool).
* Augment training with more preprocessing variants.
* Try other lightweight models like EfficientNet or MobileNetV3.
* Deploy with TorchScript or ONNX for real-time use.

---

## 📜 License

This project is licensed under the MIT License.

---

Let me know if you'd like this README to be auto-filled with your actual results or if you'd like a version with LaTeX or Jupyter support!

