"""
Модуль для загрузки и предобработки датасета MNIST
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_mnist_sklearn():
    """
    Загрузка MNIST для Scikit-learn (MLP)
    Возвращает: X_train, X_test, y_train, y_test (в виде numpy массивов)
    """
    print("Загрузка MNIST из OpenML...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='pandas')

    # Преобразование типов
    X = X.astype('float32')
    y = y.astype('int64')

    # Нормализация пикселей [0, 255] -> [0, 1]
    X = X / 255.0

    # Разделение на train (60000) и test (10000)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def load_mnist_pytorch():
    """
    Загрузка MNIST для PyTorch (LeNet)
    Возвращает DataLoader'ы для train, val, test
    """
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, random_split

    # Преобразования для LeNet: вход 32x32 (исходный LeNet ожидает 32x32)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # LeNet ожидает 32x32
        transforms.ToTensor(),  # [0,255] -> [0,1] и в формат (C, H, W)
        transforms.Normalize((0.1307,), (0.3081,))  # нормализация для MNIST
    ])

    # Загрузка датасета
    print("Загрузка MNIST через torchvision...")
    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Разделение train на train (50000) и val (10000)
    train_size = 50000
    val_size = 10000
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Создание DataLoader'ов
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Размер train: {len(train_dataset)}")
    print(f"Размер val: {len(val_dataset)}")
    print(f"Размер test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


def visualize_samples(X, y, num_samples=10):
    """
    Визуализация первых num_samples изображений из датасета
    """
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.ravel()

    for i in range(num_samples):
        # Для данных из sklearn (784,) -> (28,28)
        if X[i].shape == (784,):
            img = X[i].reshape(28, 28)
        else:
            img = X[i]
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Label: {y[i]}')
        axes[i].axis('off')

    plt.suptitle('Примеры изображений из датасета MNIST', fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Визуализация сохранена в plots/samples.png")


if __name__ == "__main__":
    # Проверка загрузки
    X_train, X_test, y_train, y_test = load_mnist_sklearn()
    visualize_samples(X_train, y_train)