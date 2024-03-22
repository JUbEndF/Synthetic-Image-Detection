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
from DenoisingModel import DenoisingNetwork

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
def add_noise(image):
    noise = np.random.normal(0, 0.2, image.shape)  # Создаем случайный шум
    noisy_image = image + noise  # Добавляем шум к изображению
    return noisy_image


model = DenoisingNetwork(3, 3, 2).to(args.device)

# Определите функцию потерь и оптимизатор
criterion = nn.MSELoss().to(args.device)
#optimizer = optim.SGD(model.parameters(), lr=args.lr)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

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
    print(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {epoch_loss:.4f}')

import matplotlib.pyplot as plt

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
    