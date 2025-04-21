import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import CustomDataset
from model import google_net
from train import evaluation, train, image_retrieval


def image_classification():

    train_dataset_path = '../data/2d_dataset/train/'
    test_dataset_path = '../data/2d_dataset/test/'

    batch_size = 32
    num_workers = 16
    num_classes = 97
    num_epochs = 30
    history = {"loss": [], "acc": [], "precision": [], "recall": [], "f1": [], "balanced_acc": []}
    test_history = {"loss": [], "acc": [], "precision": [], "recall": [], "f1": [], "balanced_acc": []}
    save_model_path = '../checkpoints/3d_to_image/model'
    # 定义transform
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    # 加载数据集
    train_dataset = CustomDataset(root_dir=train_dataset_path, transform=transform)
    test_dataset = CustomDataset(root_dir=test_dataset_path, transform=transform)

    # 创建Dataloader，指定批大小、子进程数量、类别数量
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 定义模型
    # model = ResNet101Classifier(num_classes=97)
    model = google_net(num_classes)
    # 计算类别权重
    y = np.array(train_dataset.targets)
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    # 转为 tensor
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    # 如果gpu可用，
    class_weights = class_weights.to(device)
    # 训练模型
    train_start_time = time.time()
    train_losses = train(model, train_loader, class_weights, device, num_epochs, history, save_model_path)
    train_end_time = time.time()
    print("Training time: ", train_end_time - train_start_time)
    # 打印训练集的损失曲线
    plt.plot(train_losses, label='Training loss')
    plt.legend()
    plt.show()
    # 评估模型
    eval_start_time = time.time()
    evaluation(model, test_loader, save_model_path)
    eval_end_time = time.time()
    print("Evaluation time: ", eval_end_time - eval_start_time)
    image_retrieval(model, train_loader, test_loader, device)
if __name__ == '__main__':
    image_classification()



