import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CLASS_NAMES = ['glioma', 'meningioma', 'pituitary', 'notumor']
LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for label_name in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label_name)
            if os.path.isdir(label_path):
                for file in os.listdir(label_path):
                    if file.endswith(('.jpg', '.png')):
                        self.samples.append((os.path.join(label_path, file), LABEL_MAP[label_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
    
def get_transforms():
    return transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.18794892, 0.18794641, 0.18795056],
        std=[0.18724446, 0.18724621, 0.18725474]
        )
])