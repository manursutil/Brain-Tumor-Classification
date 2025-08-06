import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
import json
from sklearn.metrics import classification_report, confusion_matrix

from dataset import BrainTumorDataset, get_transforms, CLASS_NAMES
from utils import calculate_accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights_path="../models/resnet18_brain_mri.pt"
test_dir = "../data/Testing/"

def load_model(weights_path=weights_path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    model.to(device)
    return model

def evaluate_model(model, test_dir=test_dir, batch_size=32):
    ds = BrainTumorDataset(test_dir, transform=get_transforms())
    loader = DataLoader(ds, batch_size=batch_size)

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        "accuracy": float(report["accuracy"]), # type: ignore
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }

def predict_image(model, image_path):
    image = Image.open(image_path).convert("RGB")
    transform = get_transforms()
    tensor = transform(image).unsqueeze(0).to(device) # type: ignore

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred_class = torch.max(probs, dim=1)
    
    return {
        "predicted_class": CLASS_NAMES[pred_class.item()], # type: ignore
        "confidence": round(conf.item(), 4),
        "class_index": pred_class.item()
    }
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["evaluate", "predict"], required=True)
    parser.add_argument("--image", help="Path to image if mode=predict")
    parser.add_argument("--weights", default=weights_path)
    args = parser.parse_args()

    model = load_model(args.weights)

    if args.mode == "evaluate":
        result = evaluate_model(model)
        print(json.dumps(result, indent=2))
    elif args.mode == "predict":
        if not args.image:
            raise ValueError("You must provide --image when using mode=predict")
        result = predict_image(model, args.image)
        print(json.dumps(result, indent=2))