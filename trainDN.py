import os
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import auc, confusion_matrix, roc_auc_score, roc_curve
from tqdm import tqdm
from scripts.ArgumentParser import ArgumentParser
from networks.ClassifierBasedNoiseSignal.ClassifierNoiseExtraction import NoiseExtractionClassifier
from torchvision.datasets import ImageFolder
from scripts import ErrorMetrics as em
from scripts import PrintIntoFile

arg_parser = ArgumentParser()
args = arg_parser.parse_args()

def culc_confusion_matrix(train_labels, binary_predictions):
    cm = confusion_matrix(train_labels, binary_predictions)
    #print("Confusion Matrix  (Test Set):")
    #print(cm)

    # Вычисление точности каждой категории на тестовой выборке
    real_as_real = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    real_as_fake = cm[0, 1] / (cm[0, 0] + cm[0, 1])
    fake_as_fake = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    fake_as_real = cm[1, 0] / (cm[1, 0] + cm[1, 1])

    print(f"Accuracy (Real as Real): {100 * real_as_real:.2f}%, "
          f"Accuracy (Real as Fake): {100 * real_as_fake:.2f}%, "
          f"Accuracy (Fake as Fake): {100 * fake_as_fake:.2f}%, "
          f"Accuracy (Fake as Real): {100 * fake_as_real:.2f}%")
    return cm


data_transforms = transforms.ToTensor()
def load_data(pathDir: str):
    # Список для хранения загрузчиков данных
    data_loaders = list()

    # Получаем список всех элементов в директории
    all_items = os.listdir(pathDir)[0:1]

    # Создаем загрузчики данных для каждой категории и класса
    for class_name in [item for item in all_items if os.path.isdir(os.path.join(pathDir, item))]:  # добавьте все классы
        data_folder = os.path.join(pathDir, class_name)
        dataset = ImageFolder(root=data_folder, transform=data_transforms)
        data_loaders.append(dataset)

    dataset = torch.utils.data.ConcatDataset(data_loaders)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True)



train_loader = load_data(args.train_data_path)
val_loader = load_data(args.val_data_path)
# Инициализация модели

model = NoiseExtractionClassifier(in_channels=3, num_classes=2).to(args.device)  # Предполагаем, что два класса (например, настоящие и фальшивые изображения)

os.makedirs(f"./saveModel/AIDetectionModels/{type(model).__name__}/", exist_ok=True)
file = f"./saveModel/AIDetectionModels/{type(model).__name__}/{type(model).__name__}metric.txt"

AUCROC_train = []
AUCROC_test = []

#criterion = nn.CrossEntropyLoss().to(args.device)
criterion = nn.BCEWithLogitsLoss().to(args.device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Обучение модели
for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0
    train_labels = []
    predictions_probs = []
    train_predictions = []
    print(f"Epoch [{epoch+1}/{args.epochs}]")
    for batch in tqdm(train_loader):
        inputs, labels = batch
        inputs, labels = inputs.to(args.device), labels.to(args.device)

        optimizer.zero_grad()
        outputs = model(inputs)
        one_hot_labels = nn.functional.one_hot(labels, 2)
        loss = criterion(outputs, one_hot_labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

        outputs = torch.sigmoid(outputs)  # Преобразуем выходы в бинарные предсказания

        _, preds = torch.max(outputs, 1)
        
        predictions_probs.extend(outputs.detach().cpu().numpy()[:, 1])
        train_labels.extend(labels.cpu().numpy())
        train_predictions.extend(preds.detach().cpu().numpy())
    
    
    torch.save(model, f"./saveModel/AIDetectionModels/{type(model).__name__}/" + f'{type(model).__name__}_epoch_{epoch+1}.pth')

    filePrint = PrintIntoFile.set_global_output_file(file)

    epoch_loss = running_loss / len(train_loader)

    print(f"Epoch [{epoch+1}/{args.epochs}]")
    print(f"Loss: {epoch_loss / len(train_loader)}")

    cm = culc_confusion_matrix(train_labels, train_predictions)
    em.print_metrics(train_labels, train_predictions)
    em.print_metrics_and_calculate_mean_accuracy(train_labels, train_predictions)
    PrintIntoFile.restore_output(filePrint)


    print("Test data acc")
    # Вычисление матрицы ошибок на обучающей выборке
    model.eval()
    train_predictions = []
    train_labels = []
    predictions_probs = []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)  # Преобразуем выходы в бинарные предсказания
            _, preds = torch.max(outputs, 1)
            predictions_probs.extend(outputs.detach().cpu().numpy()[:, 1])
            train_labels.extend(labels.cpu().numpy())
            train_predictions.extend(preds.detach().cpu().numpy())

    filePrint = PrintIntoFile.set_global_output_file(file)

    print("Test data acc")
    cm = culc_confusion_matrix(train_labels, train_predictions)
    em.print_metrics(train_labels, train_predictions)
    em.print_metrics_and_calculate_mean_accuracy(train_labels, train_predictions)

    PrintIntoFile.restore_output(filePrint)