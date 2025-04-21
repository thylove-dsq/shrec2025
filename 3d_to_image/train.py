import os.path
import random
from collections import Counter
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, \
    precision_score, recall_score
from sklearn.neighbors import NearestNeighbors
from torch import nn, optim
from tqdm import tqdm

from utils import plot_test_and_neighbors, report_loss_acc_f1


def train(model, train_loader, class_weights, device, num_epochs, history, save_model_path):
    model = model.to(device)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # 训练设置
    all_preds, all_labels = [], []

    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels, path in tqdm(train_loader, f'Training {epoch + 1}'):
            # 数据移到 GPU
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        record_indicators(all_labels, all_preds, history, train_loader, running_loss, train_losses)
        report_loss_acc_f1(history, f'../checkpoints/3d_to_image/metrics/Training_metrics_epoch_{epoch + 1}.png')
        # # 验证过程
        # test_all_preds, test_all_labels = [], []
        # test_losses = []
        # model.eval()
        # test_loss = 0.0
        # with torch.no_grad():
        #     for inputs, labels, path in tqdm(test_loader, 'Testing'):
        #         inputs = inputs.to(device)
        #         labels = labels.to(device)
        #         outputs = model(inputs)
        #         loss = criterion(outputs, labels)
        #         test_loss += loss.item()
        #         preds = torch.argmax(outputs, dim=1)
        #         test_all_preds.extend(preds.cpu().tolist())
        #         test_all_labels.extend(labels.cpu().tolist())
        # test_loss = record_indicators(test_all_labels, test_all_preds, test_history, test_loader, test_loss, test_losses)
        # print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        # report_loss_acc_f1(test_history, f'../checkpoints/3d_to_image/metrics/Test_metrics_epoch_{epoch + 1}.png')

    # 保存模型
    torch.save(model.cpu().state_dict(), f'{save_model_path}/googlenet.pth')
    return train_losses

def record_indicators(all_labels, all_preds, history, loader, loss, losses):
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', )
    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
    loss = loss / len(loader)
    losses.append(loss)
    history["loss"].append(loss)
    history["acc"].append(acc)
    history["precision"].append(precision)
    history["recall"].append(recall)
    history["f1"].append(f1)
    history["balanced_acc"].append(balanced_accuracy)
    return loss


def evaluation(model, test_loader, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 加载模型
    state_dict = torch.load(f'{model_path}/googlenet.pth', map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels, path in tqdm(test_loader, 'Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predicted.cpu().tolist())
    print(f"\nTest Accuracy: {100 * correct / total:.2f}%")
    return y_true, y_pred

def extract_features(model, dataloader, device):
    model.to(device)
    model.eval()
    features_by_label = {}
    with torch.no_grad():
        for inputs, labels, paths in tqdm(dataloader, desc="Extracting Features"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            features = model.extract_features(inputs)
            for i in range(len(labels)):
                label = int(labels[i].item())
                feature_vector = features[i].cpu().numpy()
                path = paths[i]
                if label not in features_by_label:
                    features_by_label[label] = []
                features_by_label[label].append((feature_vector, path))
    return features_by_label
# 使用KNN分类
def image_retrieval(model, train_loader, test_loader, device):
    train_features_by_label = extract_features(model, train_loader, device)
    test_features_by_label = extract_features(model, test_loader, device)
    train_data = [(label, feature, path) for label, features in train_features_by_label.items() for feature, path in
                  features]
    test_data = [(label, feature, path) for label, features in test_features_by_label.items() for feature, path in
                 features]
    train_labels, train_features, train_paths = zip(*train_data)
    test_labels, test_features, test_paths = zip(*test_data)
    train_labels = np.array(train_labels)
    train_features = np.array(train_features)
    train_paths = np.array(train_paths)
    test_labels = np.array(test_labels)
    test_features = np.array(test_features)
    test_paths = np.array(test_paths)
    k = 24
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(train_features)
    distances, indices = neigh.kneighbors(test_features)
    nearest_neighbor_labels = train_labels[indices]
    accuracy = np.sum(nearest_neighbor_labels[:, 0] == test_labels) / len(test_labels) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    num_visualizations = 2
    for i in range(num_visualizations):
        random_test_index = random.randint(0, len(test_features) - 1)
        nearest_neighbor_indices = indices[random_test_index]
        test_feature, test_path = test_features[random_test_index], test_paths[random_test_index]
        neighbor_labels = train_labels[nearest_neighbor_indices]
        neighbor_paths = train_paths[nearest_neighbor_indices]
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