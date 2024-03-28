import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from torchvision.transforms import transforms
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from scripts.DatasetCustom import CustomDataset
from scripts.fft import dftImage
from networks.AIDetection import AIDetectionClassifier, AIDetectionClassifierLeakyReLU
from scripts.ArgumentParser import ArgumentParser
from torchvision.datasets import ImageFolder


arg_parser = ArgumentParser()
args = arg_parser.parse_args()

# Определение размера входных данных
input_size = 256 * 3  # Предполагается, что изображения имеют три канала и размер 256x256

# Определение размеров скрытых слоев
hidden_sizes = [128, 64, 32, 16, 8]

# Определение числа классов
num_classes = 2

# Определение трансформации данных
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_file = 'train_dataset.pt'
test_file = 'test_dataset.pt'

if os.path.exists(train_file):
    train_dataset = torch.load(train_file)
else:
    train_dataset = CustomDataset(args.train_data_path, args.device)
    torch.save(train_dataset, train_file)

if os.path.exists(test_file):
    test_dataset = torch.load(test_file)
else:
    test_dataset = CustomDataset(args.test_data_path, args.device)
    torch.save(test_dataset, test_file)

# Создание DataLoader для обучения
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)



# Инициализация модели
#model = AIDetectionClassifier(input_size, hidden_sizes).to(args.device)
model = AIDetectionClassifier(input_size, hidden_sizes, 1).to(args.device)

# Определение функции потерь и оптимизатора
criterion = nn.BCEWithLogitsLoss().to(args.device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

# Обучение модели
for epoch in range(args.epochs):
    print('-' * 40)
    model.train()
    print(epoch+1)
    running_loss = 0.0
    train_labels = []
    train_predictions = []
    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(args.device).float(), labels.to(args.device).float()
        inputs = inputs.view(inputs.size(0), -1)  # Выпрямляем изображения в векторы
        optimizer.zero_grad()
        #print(inputs.shape)
        outputs = model(inputs)
        outputs = outputs.view(-1)  # Преобразовать в одномерный тензор
        loss = criterion(outputs, labels) #criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        outputs = outputs.view(-1)
        train_labels.extend(labels.cpu().numpy())
        train_predictions.extend(outputs.detach().cpu().numpy())

    # Вычисление матрицы ошибок
    threshold = 0.5
    train_predictions_array = np.array(train_predictions)
    binary_predictions = (train_predictions_array > threshold).astype(int)
    cm = confusion_matrix(train_labels, binary_predictions)
    print("Result train")
    print("Confusion Matrix  (Test Set):")
    print(cm)

    # Вычисление точности каждой категории на тестовой выборке
    real_as_real = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    real_as_fake = cm[0, 1] / (cm[0, 0] + cm[0, 1])
    fake_as_fake = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    fake_as_real = cm[1, 0] / (cm[1, 0] + cm[1, 1])

    print(f"Accuracy (Real as Real): {100 * real_as_real:.2f}%, "
          f"Accuracy (Real as Fake): {100 * real_as_fake:.2f}%, "
          f"Accuracy (Fake as Fake): {100 * fake_as_fake:.2f}%, "
          f"Accuracy (Fake as Real): {100 * fake_as_real:.2f}%")
    
    print(f"Epoch {epoch+1}/{args.epochs}, Loss: {running_loss / len(train_loader)}")
    

    torch.save(model, "./saveModel/AIDetectionModels/" + f'modelAIDetection_epoch_{epoch+1}.pth')

    # Оценка модели на тестовой выборке
    model.eval()
    test_labels = []
    test_predictions = []
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            images, labels = inputs.to(args.device).float(), labels.to(args.device).float()
            images = images.view(images.size(0), -1)
            outputs = model(images)
            outputs = outputs.view(-1)  # Преобразовать в одномерный тензор
            test_labels.extend(labels.cpu().numpy())
            test_predictions.extend(outputs.cpu().numpy())

    # Вычисление матрицы ошибок
    train_predictions_array = np.array(test_predictions)
    binary_predictions = (train_predictions_array > threshold).astype(int)
    cm = confusion_matrix(test_labels, binary_predictions)
    print("Result val")
    print("Confusion Matrix (Test Set):")
    print(cm)

    # Вычисление точности каждой категории на тестовой выборке
    real_as_real = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    real_as_fake = cm[0, 1] / (cm[0, 0] + cm[0, 1])
    fake_as_fake = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    fake_as_real = cm[1, 0] / (cm[1, 0] + cm[1, 1])

    print(f"Accuracy (Real as Real): {100 * real_as_real:.2f}%, "
          f"Accuracy (Real as Fake): {100 * real_as_fake:.2f}%, "
          f"Accuracy (Fake as Fake): {100 * fake_as_fake:.2f}%, "
          f"Accuracy (Fake as Real): {100 * fake_as_real:.2f}%")

