import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
import cv2
from tqdm import tqdm
from scripts.ArgumentParser import ArgumentParser
from networks.ClassifierBasedNoiseSignal.DenoisingModel import DenoisingNetwork
from torch.optim.lr_scheduler import ReduceLROnPlateau


def add_noise(image):
    """
    Добавляет случайный шум к изображению RGB.
    :param image: Изображение RGB в формате numpy array (например, (height, width, 3)).
    :return: Изображение с добавленным шумом.
    """
    # Список доступных типов шумов
    noise_types = ['gaussian', 'salt_and_pepper', 'speckle', 'poisson']

    # Случайно выбираем тип шума
    noise_type = random.choice(noise_types)

    # Добавляем выбранный шум к изображению
    if noise_type == 'gaussian':
        noisy_image = add_gaussian_noise(image)
    elif noise_type == 'salt_and_pepper':
        noisy_image = add_salt_and_pepper_noise(image)
    elif noise_type == 'speckle':
        noisy_image = add_speckle_noise(image)
    elif noise_type == 'poisson':
        noisy_image = add_poisson_noise(image)

    return noisy_image

def add_gaussian_noise(image):
    """Добавляет гауссовский шум к изображению."""
    noise = np.random.normal(0, 0.05, image.shape)  # Создаем случайный гауссовский шум
    noisy_image = image + noise  # Добавляем шум к изображению
    noisy_image = np.clip(noisy_image, 0, 1)  # Ограничиваем значения в диапазоне [0, 1]
    return noisy_image

def add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    """Добавляет шум солевого и перцового типа к изображению."""
    noisy_image = np.copy(image)
    num_salt = np.ceil(salt_prob * image.size)
    num_pepper = np.ceil(pepper_prob * image.size)

    # Добавляем солевой шум
    salt_coords = tuple([np.random.randint(0, i - 1, int(num_salt)) for i in image.shape])
    noisy_image[salt_coords] = 1

    # Добавляем перцовый шум
    pepper_coords = tuple([np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape])
    noisy_image[pepper_coords] = 0

    return noisy_image

def add_speckle_noise(image):
    """Добавляет шум пятен к изображению."""
    noise = np.random.normal(0, 0.05, image.shape)  # Создаем случайный шум
    noisy_image = image + image * noise  # Добавляем шум к изображению
    noisy_image = np.clip(noisy_image, 0, 1)  # Ограничиваем значения в диапазоне [0, 1]
    return noisy_image

def add_poisson_noise(image):
    """Добавляет пуассонов шум к изображению."""
    noisy_image = np.random.poisson(image * 255) / 255  # Применяем пуассонов шум
    noisy_image = np.clip(noisy_image, 0, 1)  # Ограничиваем значения в диапазоне [0, 1]
    return noisy_image

def load_data(pathDir: str):
    # Список для хранения загрузчиков данных
    data_loaders = list()

    # Получаем список всех элементов в директории
    all_items = os.listdir(pathDir)

    transform = transforms.Compose([
        transforms.ToTensor()  # Преобразуем изображения в тензоры PyTorch
    ])

    # Создаем загрузчики данных для каждой категории и класса
    for class_name in [item for item in all_items if os.path.isdir(os.path.join(pathDir, item))]:  # добавьте все классы
        data_folder = os.path.join(pathDir, class_name)
        dataset = ImageFolder(root=data_folder, transform=transform)
        data_loaders.append(dataset)

    dataset = torch.utils.data.ConcatDataset(data_loaders)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

if __name__ == '__main__':


    arg_parser = ArgumentParser()
    args = arg_parser.parse_args()

    # Предположим, что у вас есть некоторая папка с изображениями
    # Используем torchvision для загрузки изображений и трансформации

    # Определите трансформации для загрузки изображений и их преобразования
    transform = transforms.Compose([
        transforms.ToTensor()  # Преобразуем изображения в тензоры PyTorch
    ])


    # Создаем DataLoader для загрузки данных
    folder = "D:\\archive (1)\\Google Universal Image Embedding GUIE JPGCSV"

    dataset = ImageFolder(root=folder, transform=transform)

    # Определение долей для обучающей и валидационной выборок
    train_val_split = 0.8  # 80% для обучения, 20% для валидации

    # Вычисление размеров обучающей и валидационной выборок
    train_size = int(train_val_split * len(dataset))
    val_size = len(dataset) - train_size

    # Разделение датасета на обучающую и валидационную выборки
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Создание загрузчиков данных для обучающей и валидационной выборок
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)


    model = DenoisingNetwork().to(args.device)

    # Определите функцию потерь и оптимизатор
    criterion = nn.MSELoss().to(args.device)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    

    os.makedirs(f"./saveModel//{type(model).__name__}/", exist_ok=True)
    file = f"./saveModel/{type(model).__name__}/{type(model).__name__}metric.txt"

    for epoch in range(30):
        model.train()  # Установка модели в режим обучения
        running_loss = 0.0
        print(f"Epoch {epoch +1}")
        for images, labels in tqdm(train_loader):
            # Обнуление градиентов
            optimizer.zero_grad()

            # Получаем предсказания модели для шумных изображений
            noisy_images = torch.stack([torch.tensor(add_noise(image.numpy())) for image in images]).float()
            noisy_images = noisy_images.to(args.device)
            outputs = model(noisy_images)

            # Вычисляем потери
            loss = criterion(outputs, images.to(args.device))  # Сравниваем предсказания с оригинальными изображениями

            # Обратное распространение ошибки и обновление весов
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        # Вычисляем средние потери за эпоху
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {epoch_loss}')
        
        # Оценка на валидационной выборке
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_images, val_labels in tqdm(val_loader):
                # Создаем шумные изображения
                val_noisy_images = torch.stack([torch.tensor(add_noise(val_image.numpy())) for val_image in val_images]).float()
                val_noisy_images = val_noisy_images.to(args.device)

                # Получаем выходы модели
                val_outputs = model(val_noisy_images)

                # Вычисляем потери
                val_loss += criterion(val_outputs, val_images.to(args.device)).item() * val_images.size(0)

        # Средние потери на валидационной выборке
        val_epoch_loss = val_loss / len(val_loader.dataset)
        print(f'Validation Loss: {val_epoch_loss}')
        torch.save(model, f"./saveModel/{type(model).__name__}/" + f'{type(model).__name__}_epoch_{epoch+1}_Lost_{val_epoch_loss}.pth')
        # Используйте планировщик скорости обучения (LR Scheduler)
        scheduler.step(val_epoch_loss)


    import matplotlib.pyplot as plt

    

    # Загрузка модели
    #loaded_model = DenoisingModel.load_model('Denoising_model_epoch_10.pth')

    # Установим модель в режим оценки
    model.eval()

    # Выберем несколько изображений из датасета для демонстрации
    num_samples = 5
    sample_loader = DataLoader(dataset, batch_size=num_samples, shuffle=True)

    # Получим пакет изображений
    images, _ = next(iter(sample_loader))

    # Добавим шум к выбранным изображениям
    noisy_images = torch.stack([torch.tensor(add_noise(image.numpy())) for image in images]).float()

    # Перенесем данные на GPU
    noisy_images = noisy_images.to(args.device)

    # Применим модель к шумным изображениям, чтобы получить восстановленные изображения
    restored_images = model(noisy_images)

    # Перенесем восстановленные изображения на CPU для отображения
    restored_images = restored_images.cpu()

    # Покажем оригинальные, шумные и восстановленные изображения
    for i in range(num_samples):
        plt.figure(figsize=(10, 5))

        # Оригинальное изображение
        plt.subplot(1, 3, 1)
        plt.title('Original')
        plt.imshow(images[i].permute(1, 2, 0))  # Переводим тензор в изображение и меняем порядок каналов

        # Шумное изображение
        plt.subplot(1, 3, 2)
        plt.title('Noisy')
        plt.imshow(noisy_images[i].permute(1, 2, 0).cpu().numpy())

        # Восстановленное изображение
        plt.subplot(1, 3, 3)
        plt.title('Restored')
        plt.imshow(restored_images[i].detach().numpy().transpose(1, 2, 0))

        plt.show()