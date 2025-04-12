import random

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from model import google_net
from sklearn.utils import compute_class_weight
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

from datasets import CustomDataset, read_label_map
from train_evaluation import evaluation, prediction, train, image_retrieval
from utils import print_class_distribution


def image_classification():
    split = 'train'.upper()
    with open('./config/vtk.yaml', 'r') as config_file:
        config_data = yaml.safe_load(config_file)
        dataset_path = config_data[split]['PIC_PATH']
        label_path = config_data[split]['LABEL_PATH']
    _, class_ids = read_label_map(label_path)
    class_to_idx = {}
    class_ids = class_ids.drop_duplicates()
    # 创建分类
    for i in range(len(class_ids)):
        class_to_idx['cls_' + str(i)] = 'cls_' + str(i)

    # 定义transform
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    # 加载数据集
    dataset = CustomDataset(root_dir=dataset_path, class_to_idx=class_to_idx, transform=transform, split='train')
    # 计算训练集、测试集和验证集数据量
    total_samples = len(dataset)
    train_size = int(0.6 * total_samples)  # 60% for training
    val_size = int(0.2 * total_samples)  # 20% for validation
    test_size = total_samples - train_size - val_size
    # 分割数据集
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    # 创建Dataloader，指定批大小、子进程数量、类别数量
    batch_size = 32
    num_workers = 8
    num_classes = 97
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # 打印训练集每个类别数据的数量
    print_class_distribution(train_set, "Training")
    print("\n")
    print_class_distribution(val_set, "Validation")
    print("\n")
    print_class_distribution(test_set, "Test")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 定义模型
    # model = ResNet101Classifier(num_classes=97)
    model = google_net()
    # 计算类别权重
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=np.array(np.arange(num_classes))
    )
    # 转为 tensor
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    # 如果gpu可用，
    class_weights = class_weights.to(device)

    # 训练模型
    train_losses, val_losses = train(model, train_loader, val_loader, class_weights)
    # 打印训练集与验证集的损失曲线
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    plt.show()
    # 评估模型
    y_true, y_pred = evaluation(model, test_loader, class_to_idx)
    # 打印分类的预测值和真实值做比较
    class_names = dataset.class_to_idx
    num_samples = 5
    samples = random.sample(range(len(test_set)), num_samples)
    fig, axs = plt.subplots(1, num_samples, figsize=(15, 3))
    for i, sample in enumerate(samples):
        image, label, path = test_set[sample]
        image = np.transpose(image.numpy(), (1, 2, 0))
        axs[i].imshow(image)
        label_name1 = class_names['cls_' + str(label.item())]
        label_name2 = class_names['cls_' + str(y_pred[sample])]
        axs[i].set_title(f"True: {label_name1}\nPredicted: {label_name2}")
    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout(pad=2.0)
    plt.show()
    # 分类
    image_retrieval(model, train_loader, test_loader, device)
    # 对测试集进行分类
    split = 'test'.upper()
    with open('./config/vtk.yaml', 'r') as config_file:
        config_data = yaml.safe_load(config_file)
        dataset_path = config_data[split]['PIC_PATH']
    classification_set = CustomDataset(root_dir=dataset_path, class_to_idx=class_to_idx, transform=transform,
                                       split='test')
    classification_loader = DataLoader(classification_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    prediction(model, classification_loader, device)

if __name__ == '__main__':
    image_classification()



