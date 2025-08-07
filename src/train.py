import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

from dataset import BrainTumorDataset, get_transforms
from utils import calculate_accuracy, log_metrics_to_csv, print_metrics, set_seed

set_seed(42)

model_path = "../models/resnet18_brain_mri.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading Datasets
train_ds = BrainTumorDataset("../data/Training", transform=get_transforms())
test_ds = BrainTumorDataset("../data/Testing", transform=get_transforms())

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

# Initializing model
weights = ResNet18_Weights.DEFAULT

model = models.resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 4)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

csv_log_path = "../models/resnet18_metrics.csv"

# Training
for epoch in range(5):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/5]", leave=False)

    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct, total = calculate_accuracy(outputs, labels)
        total_correct += correct
        total_samples += total

        loop.set_postfix(loss=loss.item(), acc=100 * total_correct / total_samples)

    epoch_accuracy = 100 * total_correct / total_samples
    print_metrics(epoch, total_loss, epoch_accuracy)
    log_metrics_to_csv(csv_log_path, epoch, total_loss, epoch_accuracy)

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    loop = tqdm(test_loader, desc="Testing", leave=False)
    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        c, t = calculate_accuracy(outputs, labels)
        correct += c
        total += t

print(f"\nTest Accuracy: {correct / total:.2%}")

torch.save(model.state_dict(), model_path)
print(f"ResNet model saved to {model_path}")
