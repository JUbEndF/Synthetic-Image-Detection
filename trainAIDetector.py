import cv2
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
from networks.AIDetection import AIDetectionClassifier
from scripts.ArgumentParser import ArgumentParser
from torchvision.datasets import ImageFolder


arg_parser = ArgumentParser()
args = arg_parser.parse_args()

# Определение размера входных данных
input_size = 256 * 3  # Предполагается, что изображения имеют три канала и размер 256x256

# Определение размеров скрытых слоев
hidden_sizes = [512, 256, 128, 64, 32]

# Определение числа классов
num_classes = 2

# Определение трансформации данных
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Загрузка обучающего набора данных без преобразования данных
train_dataset = CustomDataset(args.train_data_path, args.device)
val_dataset = CustomDataset(args.val_data_path, args.device)
test_dataset = CustomDataset(args.test_data_path, args.device)

# Создание DataLoader для обучения
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

# Инициализация модели
model = AIDetectionClassifier(input_size, hidden_sizes, num_classes).to(args.device)

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Обучение модели
for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0
    for batch in tqdm(train_loader):
        inputs, labels = batch
        inputs, labels = inputs.to(args.device).float(), labels.to(args.device)
        inputs = inputs.view(inputs.size(0), -1)  # Выпрямляем изображения в векторы
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.long()) #criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{args.epochs}, Loss: {running_loss / len(train_loader)}")
