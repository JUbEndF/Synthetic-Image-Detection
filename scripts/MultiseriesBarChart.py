import re
import matplotlib.pyplot as plt
import os
import csv

def parse_file(file_path):
    model_name = os.path.splitext(os.path.basename(file_path))[0]
    model_data = {}
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        current_dataset = None
        
        for line in lines:
            # Определение названия набора данных
            dataset_match = re.match(r'Test data acc (\w+)', line)
            if dataset_match:
                current_dataset = dataset_match.group(1)
                if current_dataset not in model_data:
                    model_data[current_dataset] = {}
            
            # Извлечение данных метрик
            if current_dataset:
                accuracy_match = re.match(r'accuracy: ([\d\.]+)', line)
                if accuracy_match:
                    model_data[current_dataset][model_name] = float(accuracy_match.group(1))
                    
    return model_data

def parse_csv(file_path):
    csv_data = {}
    name = os.path.splitext(os.path.basename(file_path))[0]
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            testset = row['Testset']
            accuracy = float(row['Acc (224)'])
            if testset not in csv_data:
                csv_data[testset] = {}
            csv_data[testset][name] = accuracy
    
    return csv_data

def plot_metrics(all_data):
    datasets = sorted(all_data.keys())
    models = sorted(next(iter(all_data.values())).keys())
    
    x = range(len(datasets))  # Индексы наборов данных
    width = 0.15  # Ширина столбцов
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for i, model in enumerate(models):
        accuracies = [all_data[dataset].get(model, 0) for dataset in datasets]
        ax.bar([p + i * width for p in x], accuracies, width=width, label=model)
    
    for i, model in enumerate(models):
        total_accuracy = 0
        count = len(datasets)
        for dataset in datasets:
            total_accuracy += all_data[dataset].get(model, 0)
        average_accuracy = total_accuracy / count
        ax.legend(title=f"{model} ({average_accuracy:.2f})", fontsize=12)

    ax.set_xlabel('Генеративные модели', fontsize=14)
    ax.set_ylabel('Точность', fontsize=14)
    ax.set_title('Точность моделей для разных искусственных изображений', fontsize=16)
    ax.set_xticks([p + width for p in x])
    ax.set_xticklabels(datasets, fontsize=12, rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.set_ylim(0.4, 1.0)
    plt.tight_layout()
    plt.show()

def main():
    # Пути к файлам
    file_paths = []#["./saveModel/AIDetectionModels/AIDetectionCNN/AIDetectionCNN.txt", './saveModel/AIDetectionModels/AIDetectionCNN7BatchNorm/AIDetectionCNN7BatchNorm.txt', './saveModel/AIDetectionModels/AIDetectionCNNBaseNormBatch/AIDetectionCNNBaseNormBatch.txt']
    #file_paths = ["./resultTrain/ClassifierNE.txt", "./resultTrain/ClassifierNEmoreL.txt", "./resultTrain/ClassifierNEPReLU.txt"]
    #file_paths = ["./resultTrain/ClassifierViT.txt", "./resultTrain/Classifierclip_vit_base_patch32.txt", "./resultTrain/ClassifierResNet.txt"]
    #file_paths = ["./resultTrain/ClassifierNE.txt", "./saveModel/AIDetectionModels/AIDetectionCNN/AIDetectionCNN.txt", "./resultTrain/Classifierclip_vit_base_patch32.txt"]
    csv_path = "./data/CNN from Paper.csv"
    csv_path2 = ''#"./data/CLIP paper.csv"
    
    # Парсинг файлов и объединение данных
    all_data = {}
    for file_path in file_paths:
        model_data = parse_file(file_path)
        for dataset, data in model_data.items():
            if dataset not in all_data:
                all_data[dataset] = {}
            all_data[dataset].update(data)
            
    if csv_path != "":
        # Парсинг данных из CSV и их добавление
        csv_data = parse_csv(csv_path)
        for dataset, data in csv_data.items():
            if dataset not in all_data:
                all_data[dataset] = {}
            all_data[dataset].update(data)
    
    if csv_path2 != "":
        # Парсинг данных из CSV и их добавление
        csv_data = parse_csv(csv_path2)
        for dataset, data in csv_data.items():
            if dataset not in all_data:
                all_data[dataset] = {}
            all_data[dataset].update(data)

    models = sorted(next(iter(all_data.values())).keys())
    for i, model in enumerate(models):
        total_accuracy = 0
        count = len(all_data)
        for dataset in all_data:
            total_accuracy += all_data[dataset].get(model, 0)
        average_accuracy = total_accuracy / count
        
        # Замена названия модели на "Модель" + средняя точность
        new_model_name = f"{model} ({average_accuracy:.2f})"
        print(new_model_name)

    # Построение графиков
    plot_metrics(all_data)

if __name__ == "__main__":
    main()