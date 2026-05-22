"""
Сравнение результатов MLP и LeNet
"""

import matplotlib.pyplot as plt
import numpy as np

def compare_models(mlp_metrics, lenet_metrics, mlp_time, lenet_time):
    """
    Сравнение метрик двух моделей
    """
    print("\n" + "="*60)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ: MLP vs LeNet")
    print("="*60)

    # Таблица сравнения
    print("\n{:<20} {:>15} {:>15} {:>15}".format("Метрика", "MLP", "LeNet", "Разница"))
    print("-" * 65)

    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1']

    for name in metrics_names:
        mlp_val = mlp_metrics[name.lower()]
        lenet_val = lenet_metrics[name.lower()]
        diff = lenet_val - mlp_val
        arrow = "↑" if diff > 0 else "↓"
        print("{:<20} {:>14.4f} {:>14.4f} {:>12.4f} {}".format(
            name, mlp_val, lenet_val, diff, arrow
        ))

    print("-" * 65)
    print("{:<20} {:>14.2f} {:>14.2f} {:>12.2f} {}".format(
        "Время обучения (сек)", mlp_time, lenet_time, lenet_time - mlp_time,
        "↑" if lenet_time > mlp_time else "↓"
    ))

    # Сравнение сложности моделей
    print("\n" + "="*60)
    print("СРАВНЕНИЕ СЛОЖНОСТИ МОДЕЛЕЙ")
    print("="*60)

    # Оценка сложности
    complexity = {
        'MLP': {'params': '~100K-300K', 'flops': 'Низкие', 'memory': 'Низкая'},
        'LeNet': {'params': '~60K', 'flops': 'Средние', 'memory': 'Средняя'}
    }

    print("\n{:<15} {:>15} {:>15} {:>15}".format("Модель", "Параметры", "FLOPs", "Память"))
    print("-" * 60)
    for model, comp in complexity.items():
        print("{:<15} {:>15} {:>15} {:>15}".format(model, comp['params'], comp['flops'], comp['memory']))

def plot_comparison(mlp_metrics, lenet_metrics):
    """
    Визуализация сравнения метрик
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    mlp_values = [mlp_metrics[m.lower()] for m in metrics]
    lenet_values = [lenet_metrics[m.lower()] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, mlp_values, width, label='MLP', color='skyblue')
    bars2 = ax.bar(x + width/2, lenet_values, width, label='LeNet', color='lightcoral')

    ax.set_ylabel('Значение')
    ax.set_title('Сравнение метрик: MLP vs LeNet')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    # Добавление значений на столбцы
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    ax.set_ylim(0.95, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('plots/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_error_analysis(mlp_cm, lenet_cm):
    """
    Анализ ошибок моделей
    """
    # Находим классы, которые модели часто путают
    mlp_errors = mlp_cm.copy()
    lenet_errors = lenet_cm.copy()

    # Убираем диагональ (правильные предсказания)
    for i in range(10):
        mlp_errors[i, i] = 0
        lenet_errors[i, i] = 0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # MLP ошибки
    im1 = axes[0].imshow(mlp_errors, cmap='Reds')
    axes[0].set_title('Ошибки MLP (недиагональные элементы)')
    axes[0].set_xlabel('Предсказанный класс')
    axes[0].set_ylabel('Истинный класс')
    plt.colorbar(im1, ax=axes[0])

    # Добавление значений
    for i in range(10):
        for j in range(10):
            if mlp_errors[i, j] > 0:
                axes[0].text(j, i, str(mlp_errors[i, j]), ha='center', va='center')

    # LeNet ошибки
    im2 = axes[1].imshow(lenet_errors, cmap='Reds')
    axes[1].set_title('Ошибки LeNet (недиагональные элементы)')
    axes[1].set_xlabel('Предсказанный класс')
    axes[1].set_ylabel('Истинный класс')
    plt.colorbar(im2, ax=axes[1])

    for i in range(10):
        for j in range(10):
            if lenet_errors[i, j] > 0:
                axes[1].text(j, i, str(lenet_errors[i, j]), ha='center', va='center')

    plt.suptitle('Сравнение ошибок классификации', fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/error_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Здесь нужно загрузить реальные метрики из предыдущих частей
    # Пример для демонстрации:

    mlp_metrics = {
        'accuracy': 0.9785,
        'precision': 0.9784,
        'recall': 0.9785,
        'f1': 0.9784
    }

    lenet_metrics = {
        'accuracy': 0.9905,
        'precision': 0.9905,
        'recall': 0.9905,
        'f1': 0.9905
    }

    mlp_train_time = 45.2
    lenet_train_time = 168.5

    # Сравнение
    compare_models(mlp_metrics, lenet_metrics, mlp_train_time, lenet_train_time)
    plot_comparison(mlp_metrics, lenet_metrics)