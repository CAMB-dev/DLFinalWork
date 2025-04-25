import cv2
import random
import cupy as cp
import os
import pickle
import numpy as np


def load_cifar10_batch(batch_file):
    """加载一个CIFAR-10批次"""
    with open(batch_file, 'rb') as f:
        batch_data = pickle.load(f, encoding='bytes')
    # 获取图像数据和标签
    data = batch_data[b'data']
    labels = batch_data[b'labels']

    # 将数据重新格式化为图片格式（N x 3 x 32 x 32）
    data = data.reshape((len(data), 3, 32, 32))
    data = data.astype(np.float32)

    return data, np.array(labels)


def load_cifar10_data(data_dir):
    """加载CIFAR-10训练集和测试集"""
    # 加载训练数据（CIFAR-10由5个训练batch组成）
    x_train = []
    y_train = []

    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f"data_batch_{i}")
        data, labels = load_cifar10_batch(batch_file)
        x_train.append(data)
        y_train.append(labels)

    # 将所有训练批次的数据合并成一个大数组
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    # 加载测试数据
    test_batch_file = os.path.join(data_dir, "test_batch")
    x_test, y_test = load_cifar10_batch(test_batch_file)

    return x_train, y_train, x_test, y_test


def one_hot_encode(labels, num_classes=10):
    """将标签转换为独热编码格式"""
    one_hot_labels = np.zeros((labels.shape[0], num_classes))
    one_hot_labels[np.arange(labels.shape[0]), labels] = 1
    return one_hot_labels


def load_and_process_cifar10(data_dir):
    """加载CIFAR-10数据集，并进行预处理"""
    x_train, y_train, x_test, y_test = load_cifar10_data(data_dir)

    # 将数据归一化到[0, 1]之间
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # 转换标签为独热编码
    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)

    return x_train, y_train, x_test, y_test


def to_xp(x):
    """NumPy -> CuPy"""
    return cp.asarray(x)


def apply_transform_batch(batch_imgs, transform):
    return cp.stack([transform(img) for img in batch_imgs])


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return img[:, :, ::-1]  # (C, H, W)
        return img


class RandomCrop:
    def __init__(self, size=32, padding=4):
        self.size = size
        self.padding = padding

    def __call__(self, img):
        c, h, w = img.shape
        pad = self.padding
        img_padded = cp.pad(
            img, ((0, 0), (pad, pad), (pad, pad)), mode='constant')

        top = random.randint(0, 2 * pad)
        left = random.randint(0, 2 * pad)
        return img_padded[:, top:top + self.size, left:left + self.size]


class Normalize:
    def __init__(self, mean, std):
        self.mean = cp.array(mean).reshape(3, 1, 1)
        self.std = cp.array(std).reshape(3, 1, 1)

    def __call__(self, img):
        return (img - self.mean) / self.std


class ToTensor:
    def __call__(self, img):
        return cp.asarray(img / 255.0, dtype=cp.float32)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


def load_cats_dogs(data_dir, img_size=(64, 64)):
    categories = ['cat', 'dog']
    x_data = []
    y_data = []
    for label, category in enumerate(categories):
        folder = os.path.join(data_dir, category)
        for img_file in os.listdir(folder):
            img_path = os.path.join(folder, img_file)
            try:
                # 检查文件是否为图像（防止非图像文件干扰）
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                img = cv2.imread(img_path)
                if img is None:
                    # print(f"Warning: Failed to read {img_path}, skipping.")
                    continue
                # 处理灰度图像（强制转换为3通道）
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                img = cv2.resize(img, img_size)  # 确保 img_size 有效（如 (64,64)）
                img = img.transpose(2, 0, 1)     # (H, W, C) -> (C, H, W)
                x_data.append(img)
                y_data.append(label)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    if len(x_data) == 0:
        raise ValueError("No valid images found in the dataset directory.")
    x_data = np.array(x_data, dtype=np.float32) / 255.0  # 归一化
    y_data = np.array(y_data)
    return x_data, y_data


def load_and_process_cats_dogs(data_dir):
    x_data, y_data = load_cats_dogs(data_dir)
    # 划分训练集和测试集（示例比例）
    split = int(0.8 * len(x_data))
    x_train, y_train = x_data[:split], y_data[:split]
    x_test, y_test = x_data[split:], y_data[split:]

    # shuffle
    indices = np.arange(len(x_data))
    np.random.shuffle(indices)
    x_data = x_data[indices]
    y_data = y_data[indices]

    # 转换为独热编码（二分类）
    y_train = one_hot_encode(y_train, num_classes=2)
    y_test = one_hot_encode(y_test, num_classes=2)
    return x_train, y_train, x_test, y_test
