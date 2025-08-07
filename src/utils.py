import csv
import os
import random

import numpy as np
import torch


def calculate_accuracy(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct, total

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
def print_metrics(epoch, loss, accuracy):
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
    
def log_metrics_to_csv(csv_path, epoch, loss, accuracy):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['epoch', 'loss', 'accuracy'])
        writer.writerow([epoch + 1, round(loss, 4), round(accuracy, 2)])