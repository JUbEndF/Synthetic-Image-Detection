import os
from matplotlib import transforms
from sklearn.metrics import confusion_matrix
import torch
from scripts.ArgumentParser import ArgumentParser
from networks.modelCNN.AIDetectionCNNModel import AIDetectionCNN
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


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
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

data_transforms = transforms.ToTensor()

testFolder = "./data/CNN_synth_testset/"

model_path = "./saveModel/AIDetectionModels/AIDetectionCNN/AIDetectionCNN_epoch_1.pth"
model = torch.load(model_path)

print(type(model))

#for datatype in [item for item in os.listdir(testFolder) if os.path.isdir(os.path.join(testFolder, item))]:


