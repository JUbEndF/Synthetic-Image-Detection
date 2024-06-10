import os
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
import torch
from tqdm import tqdm
from scripts.ArgumentParser import ArgumentParser
from networks.ClassifierBasedNoiseSignal.ClassifierNoiseExtraction import (
    DenoisingNetwork, NoiseExtractionClassifier2
)
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
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

def check_directory_structure(directory_path):
    # Получите список всех поддиректорий в указанной директории
    subdirs = [subdir for subdir in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, subdir))]

    # Список папок, которые мы ожидаем найти в директории
    expected_subdirs = {"0_real", "1_fake"}

    # Преобразуйте список поддиректорий в набор (set)
    subdirs_set = set(subdirs)

    # Проверьте, что набор поддиректорий совпадает с ожидаемым набором
    if subdirs_set == expected_subdirs:
        print(f"Папка '{directory_path}' содержит только папки '0_real' и '1_fake'.")
        return True
    else:
        print(f"Папка '{directory_path}' содержит дополнительные папки или отсутствуют папки '0_real' и '1_fake'.")
        return False

def load_data(pathDir: str):
    # Список для хранения загрузчиков данных
    data_loaders = list()

    # Получаем список всех элементов в директории
    all_items = os.listdir(pathDir)

    # Создаем загрузчики данных для каждой категории и класса
    for class_name in [item for item in all_items if os.path.isdir(os.path.join(pathDir, item))]:
        data_folder = os.path.join(pathDir, class_name)
        dataset = ImageFolder(root=data_folder, transform=data_transforms)
        data_loaders.append(dataset)

    dataset = torch.utils.data.ConcatDataset(data_loaders)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

if __name__ == '__main__':
    data_transforms = transforms.Compose([
        transforms.CenterCrop(size=(256, 256)),
        transforms.ToTensor()  # Преобразуем изображения в тензоры PyTorch
    ])
    testFolder = "./data/CNN_synth_testset/"
    denoising_network = DenoisingNetwork().to('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "./saveModel/ClassifierNEPReLU_epoch_10.pth"
    model = torch.load(model_path)

    print(type(model))

    for datatype in [item for item in os.listdir(testFolder) if os.path.isdir(os.path.join(testFolder, item))]:
        folder = os.path.isdir(os.path.join(testFolder, datatype))
        if not folder:
            continue
        folder = os.path.join(testFolder, datatype)
        if check_directory_structure(folder):
            dataset = ImageFolder(root=folder, transform=data_transforms)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        else:
            dataloader = load_data(folder)

        file = f"./resultTrain/{type(model).__name__}testmetrix.txt"
        model.eval()
        train_predictions = []
        train_labels = []
        predictions_probs = []
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader):
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                inputs = inputs - denoising_network(inputs)
                outputs = model(inputs)
                outputs = torch.sigmoid(outputs)  # Преобразуем выходы в бинарные предсказания
                _, preds = torch.max(outputs, 1)
                predictions_probs.extend(outputs.detach().cpu().numpy()[:, 1])
                train_labels.extend(labels.cpu().numpy())
                train_predictions.extend(preds.detach().cpu().numpy())

        filePrint = PrintIntoFile.set_global_output_file(file)

        print(f"Test data acc {datatype} ")
        cm = culc_confusion_matrix(train_labels, train_predictions)
        em.print_metrics(train_labels, train_predictions)
        em.print_metrics_and_calculate_mean_accuracy(train_labels, train_predictions, predictions_probs)

        PrintIntoFile.restore_output(filePrint)