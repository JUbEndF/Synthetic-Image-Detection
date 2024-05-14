import os
import numpy as np
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
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
from tqdm import tqdm
from scripts.ArgumentParser import ArgumentParser
from networks.modelCLIP.AIDetectionCLIPModel import( 
     ImprovedClassifierCLIP12LBatchNorm1d, 
     ImprovedClassifierCLIP12L,
     ImprovedClassifierCLIP12Lkaiming_normal, 
     ImprovedClassifierCLIP40L, 
     DeepCLIPClassifier,
     DeepClassifierCLIP16384
    )
from torchvision.datasets import ImageFolder
from scripts import ErrorMetrics as em
from scripts import PrintIntoFile
from torch.optim import lr_scheduler
from scripts.LoadModelForTrain import loadModel


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

def loadDataCLIPFILE(file_path):

    # Загрузка данных из файла с помощью NumPy
    data = np.load(file_path)

    # Получите векторы признаков и метки классов
    features = data["features"]
    labels = data["labels"]

    # Преобразуйте векторы признаков и метки классов в тензоры PyTorch
    features_tensor = torch.tensor(features)
    labels_tensor = torch.tensor(labels)

    dataset = TensorDataset(features_tensor, labels_tensor)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Теперь у вас есть DataLoader, который вы можете использовать для обучения или оценки модели
    print(f"DataLoader успешно создан. Размер данных: {len(dataset)}")
    return dataloader
def loadDataCLIPFILE(file_path):

    # Загрузка данных из файла с помощью NumPy
    data = np.load(file_path)

    # Получите векторы признаков и метки классов
    features = data["features"]
    labels = data["labels"]

    # Преобразуйте векторы признаков и метки классов в тензоры PyTorch
    features_tensor = torch.tensor(features)
    labels_tensor = torch.tensor(labels)

    dataset = TensorDataset(features_tensor, labels_tensor)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Теперь у вас есть DataLoader, который вы можете использовать для обучения или оценки модели
    print(f"DataLoader успешно создан. Размер данных: {len(dataset)}")
    return dataloader

def load_data(pathDir: str):
    # Список для хранения загрузчиков данных
    data_loaders = list()

    # Получаем список всех элементов в директории
    all_items = os.listdir(pathDir)

    # Создаем загрузчики данных для каждой категории и класса
    for class_name in [item for item in all_items if os.path.isdir(os.path.join(pathDir, item))]:  # добавьте все классы
        data_folder = os.path.join(pathDir, class_name)
        dataset = ImageFolder(root=data_folder, transform=data_transforms)
        data_loaders.append(dataset)

    dataset = torch.utils.data.ConcatDataset(data_loaders)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    args = arg_parser.parse_args()

    data_transforms = transforms.ToTensor()

    if True:
        train_loader = loadDataCLIPFILE("clip_features_train.npz")
        val_loader = loadDataCLIPFILE("clip_features_val.npz")
    else:
        train_loader = load_data(args.train_data_path)
        val_loader = load_data(args.val_data_path)

    for models in [DeepClassifierCLIP16384, 
                   ImprovedClassifierCLIP12LBatchNorm1d, 
                   ImprovedClassifierCLIP12L, 
                   ImprovedClassifierCLIP12Lkaiming_normal, 
                    ImprovedClassifierCLIP40L, 
                    DeepCLIPClassifier,]:
        model = models(model_name="google/vit-base-patch16-224", feature_dim=768).to(args.device)
        
        os.makedirs(f"./saveModel/AIDetectionModels/{type(model).__name__}/", exist_ok=True)
        file = f"./saveModel/AIDetectionModels/{type(model).__name__}/{type(model).__name__}metric.txt"

        start, model = loadModel(model, args.device)

        AUCROC_train = []
        AUCROC_test = []
        AUCROC_train = []
        AUCROC_test = []

        criterion = nn.BCELoss().to(args.device)
        momentum = 0.9

        # Оптимизатор SGD
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=momentum)
        step_size = 7
        gamma = 0.1
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


        # Обучение модели
        for epoch in range(start, args.epochs):
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
                outputs = torch.sigmoid(outputs)  # Преобразуем выходы в вероятности
                one_hot_labels = nn.functional.one_hot(labels, 2).float()
                loss = criterion(outputs, one_hot_labels)  # Используем BCELoss с выходами модели и метками
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                _, preds = torch.max(outputs, 1)

                predictions_probs.extend(outputs.detach().cpu().numpy()[:, 1])
                train_labels.extend(labels.cpu().numpy())
                train_predictions.extend(preds.detach().cpu().numpy())
            
            
            torch.save(model, f"./saveModel/AIDetectionModels/{type(model).__name__}/" + f'{type(model).__name__}_epoch_{epoch+1}.pth')
            
            epoch_loss = running_loss / len(train_loader)

            print(f"Loss: {epoch_loss / len(train_loader)}")
            filePrint = PrintIntoFile.set_global_output_file(file)

            

            print(f"Epoch [{epoch+1}/{args.epochs}]")
            print(f"Loss: {epoch_loss / len(train_loader)}")
            print(f"Epoch [{epoch+1}/{args.epochs}]")
            print(f"Loss: {epoch_loss / len(train_loader)}")

            cm = culc_confusion_matrix(train_labels, train_predictions)
            em.print_metrics(train_labels, train_predictions)
            em.print_metrics_and_calculate_mean_accuracy(train_labels, train_predictions, predictions_probs)
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
            filePrint = PrintIntoFile.set_global_output_file(file)

            print("Test data acc")
            cm = culc_confusion_matrix(train_labels, train_predictions)
            em.print_metrics(train_labels, train_predictions)
            em.print_metrics_and_calculate_mean_accuracy(train_labels, train_predictions, predictions_probs)

            PrintIntoFile.restore_output(filePrint)
            
            kappa = average_precision_score(train_labels, predictions_probs)
            print("average_precision_score: ", kappa)
            # Обновление скорости обучения в зависимости от точности на валидационной выборке
            scheduler.step(kappa)
            for param_group in optimizer.param_groups:
                print(f'Learning Rate: {param_group["lr"]}')
