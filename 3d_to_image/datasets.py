import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


# 读csv中的文件名称和所对应的类别
def read_label_map(label_map_path):
    data = pd.read_csv(label_map_path)
    if 'train' in label_map_path:
        protein_ids = data['protein_id']
        class_ids = data['class_id']
    else:
        protein_ids = data['test_prot_id']
        class_ids = data['ground_truth_class']
    return protein_ids, class_ids

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples, self.targets = self.get_sample(root_dir)
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(int(label), dtype=torch.long), path
    def get_sample(self, path):
        samples = []
        targets = []
        for class_name in os.listdir(path):
            class_path = os.path.join(path, class_name)
            if not os.path.isdir(class_path):
                continue
            for instance in os.listdir(class_path):
                instance_path = os.path.join(class_path, instance)
                for fname in os.listdir(instance_path):
                    fpath = os.path.join(instance_path, fname)
                    label = class_name
                    samples.append((fpath, label))
                    targets.append(label)
        return samples, targets
