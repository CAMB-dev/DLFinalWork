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
