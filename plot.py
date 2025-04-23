from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_loss(train_losses, save_path="loss_curve.png"):
    plt.figure()
    plt.plot(train_losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title("Loss Plot")
    plt.savefig(save_path)
    plt.close()


# 绘制混淆矩阵的函数
def plot_confusion_matrix(y_true, y_pred, class_names, filename="confusion_matrix.png"):
    """计算并绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 正规化

    # 使用seaborn绘制热图
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()

    # 保存图像
    plt.savefig(filename)
    plt.close()

# 绘制训练损失和验证准确率的函数并保存为图片


def plot_metrics(train_losses, val_losses, val_accuracies, filename="metrics.png"):
    """绘制训练损失、验证损失和验证准确率，并保存为图片"""
    plt.figure(figsize=(12, 5))

    # 绘制训练损失和验证损失
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Loss per Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # 绘制验证准确率
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.title("Validation Accuracy per Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # 保存图表为文件
    plt.tight_layout()
    plt.savefig(filename)
