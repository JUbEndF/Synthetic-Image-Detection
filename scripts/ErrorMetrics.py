from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
    average_precision_score
)

def print_metrics(y_true, y_pred):
    """
    Выводит все метрики с их названиями и значениями, и рассчитывает среднюю точность.

    :param y_true: Истинные метки классов (список или 1D тензор).
    :param y_pred: Предсказанные метки классов (список или 1D тензор).

    :param y_true: Истинные метки классов (список или 1D тензор).
    :param y_pred: Предсказанные метки классов (список или 1D тензор).
    """
    # Рассчитываем матрицу ошибок
    cm = confusion_matrix(y_true, y_pred)
    # Рассчитываем матрицу ошибок
    cm = confusion_matrix(y_true, y_pred)

    # Рассчитываем метрики
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    # Словарь для хранения метрик с их названиями
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'F1_score': f1,
        'balanced_accuracy': balanced_accuracy,
        'kappa_statistic': kappa,
    }
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    # Словарь для хранения метрик с их названиями
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'F1_score': f1,
        'balanced_accuracy': balanced_accuracy,
        'kappa_statistic': kappa,
    }

    # Выводим метрики с их названиями и значениями
    print("Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

def print_metrics_and_calculate_mean_accuracy(y_true, y_pred, y_prob):
    """
    Выводит метрики точности правильных предсказаний с их названиями и значениями.
    Рассчитывает среднюю точность.

    :param y_true: Истинные метки классов (список или 1D тензор).
    :param y_pred: Предсказанные метки классов (список или 1D тензор).
    :param y_true: Истинные метки классов (список или 1D тензор).
    :param y_pred: Предсказанные метки классов (список или 1D тензор).
    """
    # Рассчитываем метрики
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

    # Выводим метрики с их названиями и значениями
    print("Metrics of accuracy of correct predictions:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Рассчитываем среднюю точность (из precision и recall)
    ap = average_precision_score(y_true, y_prob)

    print("Average Precision:", ap)