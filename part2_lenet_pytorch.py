"""
Часть 2: Классификация MNIST с помощью LeNet (PyTorch)
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report

from data_loader import load_mnist_pytorch

# Проверка наличия GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}")


class LeNet5(nn.Module):
    """
    Реализация архитектуры LeNet-5 для MNIST

    Архитектура (оригинальная LeNet-5):
    - Conv1: 6 ядер 5x5, вход 32x32 -> выход 28x28
    - AvgPool1: 2x2 -> выход 14x14
    - Conv2: 16 ядер 5x5 -> выход 10x10
    - AvgPool2: 2x2 -> выход 5x5
    - FC1: 16*5*5 = 400 -> 120
    - FC2: 120 -> 84
    - FC3: 84 -> 10 (классы)
    """

    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()

        # Первый сверточный блок
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Второй сверточный блок
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Полносвязные слои
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # После pooling: 5x5
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        # Функции активации
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Первый блок
        x = self.tanh(self.conv1(x))  # (1,32,32) -> (6,28,28)
        x = self.avgpool1(x)  # (6,28,28) -> (6,14,14)

        # Второй блок
        x = self.tanh(self.conv2(x))  # (6,14,14) -> (16,10,10)
        x = self.avgpool2(x)  # (16,10,10) -> (16,5,5)

        # Flatten
        x = x.view(x.size(0), -1)  # (16*5*5 = 400)

        # Полносвязные слои
        x = self.tanh(self.fc1(x))  # 400 -> 120
        x = self.tanh(self.fc2(x))  # 120 -> 84
        x = self.fc3(x)  # 84 -> 10 (без softmax, т.к. CrossEntropyLoss его содержит)

        return x


class LeNet5Modern(nn.Module):
    """
    Современная версия LeNet с улучшениями:
    - Batch Normalization
    - MaxPool вместо AvgPool
    - ReLU вместо Tanh
    - Dropout для регуляризации
    """

    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(LeNet5Modern, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)  # 32 -> 16 -> 8 -> 4 после pooling
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Блок 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        # Блок 2
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # Блок 3
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Полносвязные слои с dropout
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Обучение одной эпохи
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Обнуление градиентов
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Статистика
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """
    Валидация модели
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def train_lenet(model, train_loader, val_loader, epochs=20, lr=0.001):
    """
    Полный цикл обучения модели LeNet
    """
    print("\n" + "=" * 60)
    print("ЧАСТЬ 2: LeNet (PyTorch)")
    print("=" * 60)

    model = model.to(device)

    # Функция потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Для хранения истории обучения
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0
    best_model_state = None

    start_time = time.time()

    print(f"\nНачало обучения (макс. эпох: {epochs})")
    print("-" * 50)

    for epoch in range(epochs):
        # Обучение
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Валидация
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Сохранение истории
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Обновление learning rate
        scheduler.step(val_loss)

        # Сохранение лучшей модели
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        # Вывод прогресса
        print(f"Epoch {epoch + 1:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    train_time = time.time() - start_time

    # Загрузка лучшей модели
    model.load_state_dict(best_model_state)

    print("-" * 50)
    print(f"Обучение завершено за {train_time:.2f} секунд")
    print(f"Лучшая точность на валидации: {best_val_acc:.2f}%")

    return model, history, train_time


def evaluate_lenet(model, test_loader, device):
    """
    Детальная оценка модели LeNet на тестовой выборке
    """
    print("\n" + "=" * 60)
    print("ДЕТАЛЬНАЯ ОЦЕНКА LeNet МОДЕЛИ")
    print("=" * 60)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Метрики
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"\nМетрики на тестовой выборке:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-score:  {f1:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    # Матрица ошибок
    cm = confusion_matrix(all_labels, all_preds)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Матрица ошибок (LeNet)')
    axes[0].set_xlabel('Предсказанный класс')
    axes[0].set_ylabel('Истинный класс')

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues', ax=axes[1])
    axes[1].set_title('Матрица ошибок (нормализованная)')
    axes[1].set_xlabel('Предсказанный класс')
    axes[1].set_ylabel('Истинный класс')

    plt.tight_layout()
    plt.savefig('plots/lenet_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels
    }


def plot_training_history(history):
    """
    Построение графиков обучения
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # График потерь
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    axes[0].set_xlabel('Эпоха')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Функция потерь')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # График точности
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    axes[1].set_xlabel('Эпоха')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Точность')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Кривые обучения LeNet', fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/lenet_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_lenet_filters(model, layer_name='conv1'):
    """
    Визуализация сверточных фильтров
    """
    layer = getattr(model, layer_name)
    weights = layer.weight.data.cpu().numpy()

    # Нормализация для визуализации
    vmin, vmax = weights.min(), weights.max()

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()

    for i in range(min(8, weights.shape[0])):
        # Фильтр имеет форму (out_channels, in_channels, height, width)
        filter_img = weights[i, 0]  # Берем первый входной канал
        axes[i].imshow(filter_img, cmap='RdBu', vmin=-vmax, vmax=vmax)
        axes[i].set_title(f'Filter {i + 1}')
        axes[i].axis('off')

    plt.suptitle(f'Визуализация фильтров слоя {layer_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'plots/lenet_{layer_name}_filters.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Загрузка данных
    train_loader, val_loader, test_loader = load_mnist_pytorch()

    # Создание модели
    print("\nСоздание модели LeNet-5...")
    model = LeNet5(num_classes=10)
    print(model)

    # Подсчет параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nВсего параметров: {total_params:,}")
    print(f"Обучаемых параметров: {trainable_params:,}")

    # Обучение
    trained_model, history, train_time = train_lenet(
        model, train_loader, val_loader,
        epochs=20, lr=0.001
    )

    # Оценка
    metrics = evaluate_lenet(trained_model, test_loader, device)

    # Визуализация
    plot_training_history(history)
    visualize_lenet_filters(trained_model, 'conv1')
    visualize_lenet_filters(trained_model, 'conv2')

    # Сохранение модели
    torch.save(trained_model.state_dict(), 'models/lenet5_mnist.pth')
    print("\nМодель сохранена в models/lenet5_mnist.pth")

    print("\n✅ Часть 2 (LeNet) завершена!")