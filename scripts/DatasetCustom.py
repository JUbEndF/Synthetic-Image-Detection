from os.path import isdir, join
from os import listdir
import numpy as np
from torch.utils.data import Dataset
import os
import torch
from tqdm import tqdm
from scripts.fft import dftImage


# Функция загрузки данных для использования с DataLoader

def get_image_paths(folder_path):
    image_paths = []
    #print("Loadind path to file")
    for filename in os.listdir(folder_path):
        #if filename.endswith(extension):
        full_path = folder_path + "/" + filename
        image_paths.append(full_path)
    return image_paths


def LoadDataSet(folder_path):
    onlyfiles = [f for f in listdir(folder_path) if isdir(join(folder_path, f))]
    real_image_paths = get_image_paths(folder_path + "/" + onlyfiles[0])
    fake_image_paths = get_image_paths(folder_path + "/" + onlyfiles[1])


    # Создание меток классов
    real_labels = np.ones(len(real_image_paths))
    fake_labels = np.zeros(len(fake_image_paths))

    # Объединение данных и меток
    all_image_paths = real_image_paths + fake_image_paths
    all_labels = np.concatenate([real_labels, fake_labels])

    # Создание массива признаков
    features = []
    #print("Generating SVM for each file")
    for image_path in all_image_paths:
        features.append(image_path)

    X = np.array(features)
    y = all_labels
    return X, y

class CustomDataset(Dataset):
    def __init__(self, data_folder_path, device):
        self.X, self.y = LoadDataSet(data_folder_path)
        self.device = device
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image_path = self.X[idx]
        label = self.y[idx]
        image_features = dftImage(image_path, self.device)  # Применяем функцию обработки изображений
        image_features_tensor = torch.tensor(image_features)
        return image_features_tensor, label