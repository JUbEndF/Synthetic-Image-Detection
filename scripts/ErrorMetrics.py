

def calculate_accuracy(confusion_matrix):
    """Рассчитывает точность (accuracy) из матрицы ошибок."""
    TP = confusion_matrix[0][0]
    FP = confusion_matrix[0][1]
    FN = confusion_matrix[1][0]
    TN = confusion_matrix[1][1]
    total_samples = TP + FP + FN + TN
    return (TP + TN) / total_samples

def calculate_precision(confusion_matrix):
    """Calculate precision from confusion matrix."""
    TP = confusion_matrix[0][0]
    FP = confusion_matrix[0][1]
    return TP / (TP + FP) if TP + FP > 0 else 0

def calculate_recall(confusion_matrix):
    """Calculate recall from confusion matrix."""
    TP = confusion_matrix[0][0]
    FN = confusion_matrix[1][0]
    return TP / (TP + FN) if TP + FN > 0 else 0

def calculate_f1_score(confusion_matrix):
    """Calculate F1-score from confusion matrix."""
    precision = calculate_precision(confusion_matrix)
    recall = calculate_recall(confusion_matrix)
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def calculate_specificity(confusion_matrix):
    """Calculate specificity from confusion matrix."""
    FP = confusion_matrix[0][1]
    TN = confusion_matrix[1][1]
    return TN / (TN + FP) if TN + FP > 0 else 0

def calculate_balanced_accuracy(confusion_matrix):
    """Calculate balanced accuracy from confusion matrix."""
    recall = calculate_recall(confusion_matrix)
    specificity = calculate_specificity(confusion_matrix)
    return (recall + specificity) / 2

def calculate_kappa_statistic(confusion_matrix):
    """Calculate Cohen's Kappa statistic from confusion matrix."""
    TP = confusion_matrix[0][0]
    FP = confusion_matrix[0][1]
    FN = confusion_matrix[1][0]
    TN = confusion_matrix[1][1]
    total_samples = TP + TN + FP + FN
    observed_agreement = (TP + TN) / total_samples
    
    expected_agreement = (
        ((TP + FP) / total_samples) * ((TP + FN) / total_samples) +
        ((TN + FP) / total_samples) * ((TN + FN) / total_samples)
    )
    
    return (observed_agreement - expected_agreement) / (1 - expected_agreement) if 1 - expected_agreement != 0 else 0

def calculate_false_positive_rate(confusion_matrix):
    """Calculate false positive rate from confusion matrix."""
    FP = confusion_matrix[0][1]
    TN = confusion_matrix[1][1]
    return FP / (FP + TN) if FP + TN > 0 else 0

def calculate_false_negative_rate(confusion_matrix):
    """Calculate false negative rate from confusion matrix."""
    TP = confusion_matrix[0][0]
    FN = confusion_matrix[1][0]
    return FN / (FN + TP) if FN + TP > 0 else 0

def calculate_ppv(confusion_matrix):
    """Calculate positive predictive value from confusion matrix."""
    TP = confusion_matrix[0][0]
    FP = confusion_matrix[0][1]
    return TP / (TP + FP) if TP + FP > 0 else 0

def calculate_npv(confusion_matrix):
    """Calculate negative predictive value from confusion matrix."""
    FN = confusion_matrix[1][0]
    TN = confusion_matrix[1][1]
    return TN / (TN + FN) if TN + FN > 0 else 0

def print_metrics(confusion_matrix):
    """
    Выводит все метрики с их названиями и значениями, и рассчитывает среднюю точность.
    
    :param confusion_matrix: Матрица ошибок (список из 4 элементов: [TP, FP, FN, TN]).
    """
    # Словарь для хранения метрик с их названиями
    metrics = {}

    # Рассчитываем метрики
    metrics['accuracy'] = calculate_accuracy(confusion_matrix)
    metrics['precision'] = calculate_precision(confusion_matrix)
    metrics['recall'] = calculate_recall(confusion_matrix)
    metrics['F1_score'] = calculate_f1_score(confusion_matrix)
    metrics['specificity'] = calculate_specificity(confusion_matrix)
    metrics['balanced_accuracy'] = calculate_balanced_accuracy(confusion_matrix)
    metrics['kappa_statistic'] = calculate_kappa_statistic(confusion_matrix)
    metrics['false_positive_rate'] = calculate_false_positive_rate(confusion_matrix)
    metrics['false_negative_rate'] = calculate_false_negative_rate(confusion_matrix)
    metrics['positive_predictive_value'] = calculate_ppv(confusion_matrix)
    metrics['negative_predictive_value'] = calculate_npv(confusion_matrix)

    # Список для хранения значений точности (accuracy)
    accuracy_values = []

    # Выводим метрики с их названиями и значениями
    print("Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
        # Добавляем значение точности в список
        if metric_name == 'accuracy':
            accuracy_values.append(metric_value)

def print_metrics_and_calculate_mean_accuracy(confusion_matrix):
    """
    Выводит метрики точности правильных предсказаний с их названиями и значениями.
    Рассчитывает среднюю точность.

    :param confusion_matrix: Матрица ошибок (список из 4 элементов: [TP, FP, FN, TN]).
    """
    # Рассчитываем метрики
    precision = calculate_precision(confusion_matrix)
    recall = calculate_recall(confusion_matrix)
    f1_score = calculate_f1_score(confusion_matrix)

    # Список для хранения значений точности (accuracy)
    accuracy_values = [precision, recall]

    # Выводим метрики с их названиями и значениями
    print("Metrics of accuracy of correct predictions:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")

    # Рассчитываем среднюю точность из precision и recall
    mean_accuracy = sum(accuracy_values) / len(accuracy_values)

    # Возвращаем среднюю точность
    print(f"\nAverage accuracy: {mean_accuracy:.4f}")