import math
import matplotlib.pyplot as plt

# 检查数据集是否平衡，统计每组中每个类的样本数量
def print_class_distribution(dataset, name):
    try:
        class_to_idx = getattr(dataset, 'class_to_idx', None)
        if class_to_idx is None:
            class_to_idx = getattr(dataset.dataset, 'class_to_idx', None)
        if class_to_idx is None:
            class_to_idx = getattr(dataset.dataset.dataset, 'class_to_idx', None)
        if class_to_idx is None:
            class_to_idx = {}
            for i in set(label for _, label, path in dataset):
                if i not in class_to_idx:
                    class_to_idx[i] = i
    except Exception:
        raise ValueError("Unable to determine 'class_to_idx' attribute from the dataset.")

    class_counts = [0] * len(class_to_idx)
    for _, label, path in dataset:
        class_counts[label] += 1
    print(f"{name} set class distribution:")
    for i, count in enumerate(class_counts):
        print(f"Class '{class_to_idx['cls_'+ str(i)]}': {count} samples")
def load_image(label, path):
    image = plt.imread(path)
    return image, label
def plot_test_and_neighbors(test_label, test_path, neighbor_labels, neighbor_paths):
    # 加载测试样本及其最近邻居的图像
    test_image, _ = load_image(test_label, test_path)
    neighbor_images = [load_image(neighbor_label, neighbor_path)[0] for neighbor_label, neighbor_path in
                       zip(neighbor_labels, neighbor_paths)]

    # 打印有关测试样本及其最近邻居的信息
    print("Test Sample:")
    print(f"True Label: {test_label}")
    print(f"True Path: {test_path}")
    print(f"Nearest Neighbors:")
    for i, neighbor_label in enumerate(neighbor_labels):
        print(f"Neighbor {i + 1}: Predicted Label - {neighbor_label}, Predicted Path - {neighbor_paths[i]}")
    show_num = len(neighbor_images) + 1
    num_per_row = math.ceil(math.sqrt(show_num))
    plt.figure(figsize=(2 * num_per_row, 2 * num_per_row))

    plt.subplot(num_per_row, num_per_row, 1)
    plt.imshow(test_image)
    plt.title("Test Sample")
    plt.axis('off')

    for i, neighbor_image in enumerate(neighbor_images):
        plt.subplot(num_per_row, num_per_row, i + 2)
        plt.imshow(neighbor_image)
        plt.title(f"Neighbor {i + 1}")
        plt.axis('off')
    plt.show()

def report_loss_acc_f1(history, save_dir):
    subplot_num = len(history)
    col = 3
    row = math.ceil(subplot_num / col)
    keys = ["loss", "acc", "precision", "recall", "f1", "balanced_acc"]
    labels = ["Loss", "Accuracy", "Precision", "Recall", "F1", "Balanced Accuracy"]
    title = ["Loss", "Accuracy", "Precision", "Recall", "F1", "Balanced Accuracy"]
    xlabels = ["Epoch", "Epoch", "Epoch", "Epoch", "Epoch", "Epoch"]
    ylabels = ["Loss", "Accuracy", "Precision", "Recall", "F1", "Balanced Accuracy"]
    colors = ["red", "green", "blue", "yellow", "magenta", "cyan"]
    # 可视化训练指标
    plt.figure(figsize=(4 * col, 4 * row))
    for i in range(subplot_num):
        plt.subplot(row, col, i + 1)
        plt.plot(history[keys[i]], label=labels[i], color=colors[i])
        plt.title(title[i])
        plt.xlabel(xlabels[i])
        plt.ylabel(ylabels[i])
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir)
    plt.show()