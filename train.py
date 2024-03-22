import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from networks import baseModel
from scripts import ArgumentParser
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

arg_parser = ArgumentParser.ArgumentParser()
args = arg_parser.parse_args()

# Подготовка данных

# Преобразование данных для загрузчика
transform = transforms.Compose([
    transforms.ToTensor(),  # Преобразование в тензор
    #transforms.Resize(3, 256, 256)
])

# Загрузка обучающего набора данных без преобразования данных
train_dataset = ImageFolder(root=args.train_data_path, transform=transform)
val_dataset = ImageFolder(root=args.val_data_path, transform=transform)
test_dataset = ImageFolder(root=args.test_data_path, transform=transform)

# Создание DataLoader для обучения
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

# Определение функции потерь
criterion = nn.BCEWithLogitsLoss()

# Инициализация модели и оптимизатора
device = torch.device(args.device)
model = baseModel.Net(train_dataset[0][0].shape).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Обучение модели
for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0
    print(f"Epoch {epoch + 1}")
    for batch in tqdm(train_loader):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.view(len(labels), -1)
        optimizer.zero_grad()
    

        outputs = model(inputs).float()
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    # Валидация модели
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs).float()
            loss = criterion(outputs, labels.view(len(labels), -1).float())
            val_loss += loss.item() * inputs.size(0)
            
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader.dataset)
    val_loss = val_loss / len(val_loader.dataset)
    val_accuracy = correct / total
    
    print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2%}')

# Оценка модели
model.eval()

# Инициализация двумерного массива для хранения количества правильно и неправильно классифицированных изображений
confusion_matrix = [[0, 0], [0, 0]]  # confusion_matrix[i][j] хранит количество изображений класса i, предсказанных как класс j

# Валидация модели на тестовом наборе данных
with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        

        outputs = model(inputs)
        _, predicted = outputs.max(1)
        
        for i in range(len(labels)):
            confusion_matrix[labels[i]][predicted[i]] += 1

# Вычисление общего количества изображений для каждого класса
total_real = sum(confusion_matrix[0])
total_fake = sum(confusion_matrix[1])

# Вычисление точности для каждой комбинации классов
accuracy_real_as_real = confusion_matrix[0][0] / total_real if total_real != 0 else 0
accuracy_real_as_fake = confusion_matrix[0][1] / total_real if total_real != 0 else 0
accuracy_fake_as_fake = confusion_matrix[1][1] / total_fake if total_fake != 0 else 0
accuracy_fake_as_real = confusion_matrix[1][0] / total_fake if total_fake != 0 else 0

# Вывод результатов
print(f'Real as Real Accuracy: {accuracy_real_as_real:.2%}')
print(f'Real as Fake Accuracy: {accuracy_real_as_fake:.2%}')
print(f'Fake as Fake Accuracy: {accuracy_fake_as_fake:.2%}')
print(f'Fake as Real Accuracy: {accuracy_fake_as_real:.2%}')