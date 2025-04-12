import random
from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from torch import nn, optim
from tqdm import tqdm
from utils import plot_test_and_neighbors


def train(model, train_balanced_loader, val_balanced_loader, class_weights):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # 训练设置
    num_epochs = 15
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels, path in tqdm(train_balanced_loader):
            # 数据移到 GPU
            inputs = inputs.to(device)
            labels = labels.to(device)
            #
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_balanced_loader)
        train_losses.append(train_loss)
        # 验证过程
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels, path in tqdm(val_balanced_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        validation_loss = val_loss / len(val_balanced_loader)
        val_losses.append(validation_loss)
        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {validation_loss:.4f}")
    # 保存模型
    torch.save(model.cpu().state_dict(), 'googlenet.pth')
    return train_losses, val_losses
def evaluation(model, test_loader, class_to_idx):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 加载模型
    state_dict = torch.load('googlenet.pth', map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels, path in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predicted.cpu().tolist())

    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    target_names = [class_to_idx[i] for i in sorted(class_to_idx.keys())]
    print(classification_report(y_true, y_pred, target_names=target_names))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(100, 80))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return y_true, y_pred
def extract_features(model, dataloader, device):
    model.to(device)
    model.eval()

    features_by_label = {}

    with torch.no_grad():
        for inputs, labels, paths in tqdm(dataloader, desc="Extracting Features"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)  # 假设你已经删掉分类层了
            for i in range(len(labels)):
                label = int(labels[i].item())
                feature_vector = outputs[i].cpu().numpy()  # 从 GPU 拷回 CPU
                path = paths[i]

                if label not in features_by_label:
                    features_by_label[label] = []

                features_by_label[label].append((feature_vector, path))

    return features_by_label
# 使用KNN分类
def image_retrieval(model, train_loader, test_loader, device):
    # Extract features for training set
    train_features_by_label = extract_features(model, train_loader, device)

    # Print the shape of feature vectors for each class
    for label, features in train_features_by_label.items():
        print(f"Class {label}: {features[0][1]}")

    # Extract features for test set
    test_features_by_label = extract_features(model, test_loader, device)

    # Print labels and paths of the first retrieved image for demonstration
    label = next(iter(test_features_by_label))
    features, path = test_features_by_label[label][0]
    print(f"Label: {label}, Path: {path}")

    # Convert dictionaries to lists of tuples
    train_data = [(label, feature, path) for label, features in train_features_by_label.items() for feature, path in
                  features]
    test_data = [(label, feature, path) for label, features in test_features_by_label.items() for feature, path in
                 features]

    # Extract features, labels, and paths
    train_labels, train_features, train_paths = zip(*train_data)
    test_labels, test_features, test_paths = zip(*test_data)

    # Convert lists to numpy arrays
    train_labels = np.array(train_labels)
    train_features = np.array(train_features)
    train_paths = np.array(train_paths)
    test_labels = np.array(test_labels)
    test_features = np.array(test_features)
    test_paths = np.array(test_paths)

    # Create a NearestNeighbors model
    k = 24  # Number of neighbors to find
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(train_features)

    # Find the nearest neighbors for each vector in the test set
    # Distances are Euclidean distances between feature vectors
    # Indices are the index of the feature vector with the smallest distance in the train set
    distances, indices = neigh.kneighbors(test_features)

    # For each vector in the test set, get the labels and paths of the nearest neighbors
    nearest_neighbor_labels = train_labels[indices]
    # Calculate the accuracy
    accuracy = np.sum(nearest_neighbor_labels[:, 0] == test_labels) / len(test_labels) * 100
    print(f"Accuracy: {accuracy:.2f}%")

    num_visualizations = 5
    for i in range(num_visualizations):
        # Get a random test sample index
        random_test_index = random.randint(0, len(test_features) - 1)
        nearest_neighbor_indices = indices[random_test_index]

        # Get the features, labels, and paths for the test sample and its nearest neighbors
        test_feature, test_path = test_features[random_test_index], test_paths[random_test_index]
        neighbor_labels = train_labels[nearest_neighbor_indices]
        neighbor_paths = train_paths[nearest_neighbor_indices]

        # Plot the test sample and its nearest neighbors
        plot_test_and_neighbors(test_labels[random_test_index], test_path, neighbor_labels, neighbor_paths)
# 对任务进行预测
def prediction(model, classification_loader, device):
    model.to(device)
    state_dict = torch.load('googlenet.pth', map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    label_set = []
    predictions = []
    classification_labels = {}

    with torch.no_grad():
        for inputs, labels, paths in tqdm(classification_loader):
            inputs, labels = inputs.to(device), labels.to('cpu')
            label_set.extend(labels.numpy())
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
    for i in range(len(label_set)):
        label = label_set[i]
        if 'cls_' + str(label) not in classification_labels:
            classification_labels['cls_' + str(label)] = []
            classification_labels['cls_' + str(label)].append(predictions[i])
        else:
            classification_labels['cls_' + str(label)].append(predictions[i])
    for key in classification_labels:
        c = Counter(classification_labels[key]).most_common(1)[0][0]
        classification_labels[key] = c.item()


    test_set_csv = pd.read_csv('./data/test_set.csv')
    anonymised_protein_id = test_set_csv['anonymised_protein_id']
    test_set_csv_data = []
    for file_name in anonymised_protein_id:
        key = 'cls_' + str(file_name.replace('.vtk', ''))
        value = classification_labels[key]
        test_set_csv_dict = {'anonymised_protein_id': file_name, 'category': value}
        test_set_csv_data.append(test_set_csv_dict)
    df = pd.DataFrame(test_set_csv_data)
    df.to_csv('./data/category_test_set.csv', index=False)