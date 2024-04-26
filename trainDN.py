import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
import cv2
from tqdm import tqdm

from scripts.ArgumentParser import ArgumentParser
from networks.DenoisingModel import DenoisingNetwork

arg_parser = ArgumentParser()
args = arg_parser.parse_args()

# Предположим, что у вас есть некоторая папка с изображениями
# Используем torchvision для загрузки изображений и трансформации

# Определите трансформации для загрузки изображений и их преобразования
transform = transforms.Compose([
    transforms.ToTensor()  # Преобразуем изображения в тензоры PyTorch
])

# Загружаем данные из папки с изображениями с применением трансформаций
#dataset = ImageFolder(root=args.val_data_path, transform=transform)
dataset = ImageFolder(root="D:/archive (2)/dataset/train", transform=transform)

# Создаем DataLoader для загрузки данных
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# Определим функцию для добавления шума к изображению
import numpy as np
import random
import cv2  # для добавления шума солевого и перцового

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


model = DenoisingNetwork(3, 3, 4).to(args.device)

# Определите функцию потерь и оптимизатор
criterion = nn.MSELoss().to(args.device)
#optimizer = optim.SGD(model.parameters(), lr=args.lr)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

best_val = sys.float_info.max()
bestmodel = model
bestepoch = 0

model.train()  # Установка модели в режим обучения
for epoch in range(args.epochs):
    running_loss = 0.0
    print(f"Epoch {epoch +1}")
    for images, labels in tqdm(dataloader):
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
    epoch_loss = running_loss / len(dataloader.dataset)
    print(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {epoch_loss:.10f}')
    if epoch_loss < best_val:
        bestmodel = model
        bestepoch = epoch

print(f'Epoch [{bestepoch + 1}/{args.epochs}], Loss: {best_val:.10f}')

import matplotlib.pyplot as plt
bestmodel.save_model("3_blocks_bestmodel", './saveModel')
model.save_model("3_blocks", './saveModel')

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
    