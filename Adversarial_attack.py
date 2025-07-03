import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset, ConcatDataset
import torchattacks
import timm
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
import pandas as pd

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transforms
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor()
])

# Load CIFAR-100
train_data_full = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
test_data_full = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

# Split into 80/20 train/test
train_size = int(0.8 * len(train_data_full))
val_size = len(train_data_full) - train_size
train_data, test_split = random_split(train_data_full, [train_size, val_size])

# Define DataLoaders
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_split, batch_size=128, shuffle=False)

# Function to apply LHE + High-pass filter
def preprocess_image(img_tensor):
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = (img * 255).astype(np.uint8)
    img_eq = np.zeros_like(img)
    for i in range(3):
        img_eq[..., i] = cv2.equalizeHist(img[..., i])
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    img_hp = cv2.filter2D(img_eq, -1, kernel)
    img_hp = img_hp.astype(np.float32) / 255.0
    img_tensor_out = torch.from_numpy(img_hp.transpose(2, 0, 1))
    return img_tensor_out

# Generate adversarial examples
def generate_adversarial_examples(model, dataloader):
    atk_dict = {
        "fgsm": torchattacks.FGSM(model, eps=8/255),
        "pgd": torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=7),
        "bim": torchattacks.BIM(model, eps=8/255, alpha=2/255, steps=7),
        "autoattack": torchattacks.AutoAttack(model, norm='Linf', eps=8/255),
        "random": None
    }
    model.eval()
    adv_images, adv_labels, adv_types = [], [], []

    for attack_index, (attack_name, attack) in enumerate(atk_dict.items()):
        for images, labels in tqdm(dataloader, desc=f"Generating {attack_name}"):
            images, labels = images.to(device), labels.to(device)
            if attack_name == "random":
                noise = torch.randn_like(images) * 0.03
                adv = torch.clamp(images + noise, 0, 1)
            else:
                adv = attack(images, labels)
            adv_images.append(adv.cpu())
            adv_labels.append(torch.full_like(labels, attack_index + 1).cpu())

    X_adv = torch.cat(adv_images)
    y_adv = torch.cat(adv_labels)
    return X_adv, y_adv

# Preprocess dataset
preprocessed_imgs = []
preprocessed_labels = []
for img_batch, _ in tqdm(train_loader, desc="Preprocessing original train"):
    for i in range(img_batch.shape[0]):
        p_img = preprocess_image(img_batch[i])
        preprocessed_imgs.append(p_img)
        preprocessed_labels.append(0)

X_clean = torch.stack(preprocessed_imgs)
y_clean = torch.tensor(preprocessed_labels)

# Load pretrained GhostNet
model = timm.create_model('ghostnet_100', pretrained=True, num_classes=6).to(device)

# Generate adversarial dataset
ghost_dummy = timm.create_model('ghostnet_100', pretrained=True, num_classes=100).to(device)
X_adv, y_adv = generate_adversarial_examples(ghost_dummy, test_loader)

# Preprocess adversarial examples
X_adv_processed = torch.stack([preprocess_image(x) for x in tqdm(X_adv, desc="Preprocessing adv")])

# Combine datasets
X_all = torch.cat([X_clean.to(device), X_adv_processed.to(device)])
y_all = torch.cat([y_clean.to(device), y_adv.to(device)])

# Split into train/test
from sklearn.model_selection import train_test_split
train_idx, test_idx = train_test_split(np.arange(len(X_all)), test_size=0.2, stratify=y_all.cpu())
X_train, X_test = X_all[train_idx], X_all[test_idx]
y_train, y_test = y_all[train_idx], y_all[test_idx]

train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)

# Train classifier
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
scaler = GradScaler()

epochs = 30
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_dl:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
    scheduler.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_dl):.4f}")

# Evaluate
model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_dl:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(f"Accuracy on test data: {100 * correct / total:.2f}%")

# Classification report for attack type detection
class_names = ["Clean", "FGSM", "PGD", "BIM", "AutoAttack", "RandomNoise"]
label_ids = list(range(len(class_names)))

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, labels=label_ids, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Correlation Matrix
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
correlation_matrix = df_cm.corr()
plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", xticklabels=class_names, yticklabels=class_names)
plt.title("Correlation Matrix of Confusion Matrix")
plt.show()

# Show sample predictions
print("\nSample Predictions:")
for i in range(10):
    true_label = class_names[all_labels[i]]
    pred_label = class_names[all_preds[i]]
    print(f"Sample {i+1}: True = {true_label}, Predicted = {pred_label}")
