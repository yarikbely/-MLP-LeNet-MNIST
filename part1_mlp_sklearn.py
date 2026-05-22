"""
Часть 1: Классификация MNIST с помощью MLP (Scikit-learn)
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report

from data_loader import load_mnist_sklearn, visualize_samples


def train_mlp_model(X_train, y_train, X_val=None, y_val=None):
    """
    Обучение MLPClassifier с различными конфигурациями

    Архитектура MLP (выбрана после экспериментов):
    - 2 скрытых слоя по 128 и 64 нейрона
    - ReLU активация
    - Adam оптимизатор
    - Dropout через alpha (L2 регуляризация)
    """

    print("\n" + "=" * 60)
    print("ЧАСТЬ 1: MLP (Scikit-learn)")
    print("=" * 60)

    # Конфигурация модели
    configs = [
        {
            'name': 'MLP_Small',
            'hidden_layer_sizes': (64,),
            'alpha': 0.0001,
            'learning_rate_init': 0.001,
            'max_iter': 20
        },
        {
            'name': 'MLP_Medium',
            'hidden_layer_sizes': (128, 64),
            'alpha': 0.0001,
            'learning_rate_init': 0.001,
            'max_iter': 30
        },
        {
            'name': 'MLP_Large',
            'hidden_layer_sizes': (256, 128, 64),
            'alpha': 0.0001,
            'learning_rate_init': 0.001,
            'max_iter': 30
        }
    ]

    results = {}
    best_model = None
    best_accuracy = 0

    for config in configs:
        print(f"\n--- Обучение {config['name']} ---")
        print(f"Архитектура: {config['hidden_layer_sizes']}")

        # Создание модели
        mlp = MLPClassifier(
            hidden_layer_sizes=config['hidden_layer_sizes'],
            activation='relu',
            solver='adam',
            alpha=config['alpha'],
            learning_rate_init=config['learning_rate_init'],
            max_iter=config['max_iter'],
            batch_size=128,
            verbose=True,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5
        )

        # Обучение с замером времени
        start_time = time.time()
        mlp.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Предсказания
        y_train_pred = mlp.predict(X_train)
        y_test_pred = mlp.predict(X_test)

        # Метрики
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        # Сохранение результатов
        results[config['name']] = {
            'model': mlp,
            'train_time': train_time,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'n_iterations': mlp.n_iter_,
            'loss_curve': mlp.loss_curve_
        }

        print(f"  Train accuracy: {train_acc:.4f}")
        print(f"  Test accuracy: {test_acc:.4f}")
        print(f"  Training time: {train_time:.2f} сек")
        print(f"  Iterations: {mlp.n_iter_}")

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model = mlp

    return results, best_model


def evaluate_mlp(model, X_test, y_test):
    """
    Детальная оценка лучшей MLP модели
    """
    print("\n" + "=" * 60)
    print("ДЕТАЛЬНАЯ ОЦЕНКА ЛУЧШЕЙ MLP МОДЕЛИ")
    print("=" * 60)

    # Предсказания
    y_pred = model.predict(X_test)

    # Базовые метрики
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\nМетрики на тестовой выборке:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-score:  {f1:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Визуализация матрицы ошибок
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Матрица ошибок (MLP)')
    axes[0].set_xlabel('Предсказанный класс')
    axes[0].set_ylabel('Истинный класс')

    # Нормализованная матрица ошибок
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues', ax=axes[1])
    axes[1].set_title('Матрица ошибок (нормализованная)')
    axes[1].set_xlabel('Предсказанный класс')
    axes[1].set_ylabel('Истинный класс')

    plt.tight_layout()
    plt.savefig('plots/mlp_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def visualize_mlp_weights(model):
    """
    Визуализация весов первого слоя MLP
    """
    # Веса первого слоя: (784, hidden_size)
    weights = model.coefs_[0]

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    axes = axes.ravel()

    # Нормализация для визуализации
    vmin, vmax = weights.min(), weights.max()

    for i in range(min(16, weights.shape[1])):
        # Каждый столбец весов -> изображение 28x28
        weight_image = weights[:, i].reshape(28, 28)
        axes[i].imshow(weight_image, cmap='RdBu', vmin=-vmax, vmax=vmax)
        axes[i].set_title(f'Нейрон {i + 1}')
        axes[i].axis('off')

    plt.suptitle('Визуализация весов первого слоя MLP', fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/mlp_weights.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Визуализация весов сохранена в plots/mlp_weights.png")


def plot_loss_curves(results):
    """
    Построение графиков функции потерь для разных конфигураций
    """
    plt.figure(figsize=(10, 6))

    for name, res in results.items():
        if res['loss_curve']:
            plt.plot(res['loss_curve'], label=f"{name} (final loss: {res['loss_curve'][-1]:.4f})")

    plt.xlabel('Итерация')
    plt.ylabel('Loss')
    plt.title('Кривые обучения MLP моделей')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('plots/mlp_loss_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Загрузка данных
    X_train, X_test, y_train, y_test = load_mnist_sklearn()

    # Визуализация примеров
    visualize_samples(X_train, y_train)

    # Обучение MLP
    results, best_model = train_mlp_model(X_train, y_train, X_test, y_test)

    # Оценка лучшей модели
    metrics = evaluate_mlp(best_model, X_test, y_test)

    # Визуализация весов
    visualize_mlp_weights(best_model)

    # Графики потерь
    plot_loss_curves(results)

    print("\n✅ Часть 1 (MLP) завершена!")