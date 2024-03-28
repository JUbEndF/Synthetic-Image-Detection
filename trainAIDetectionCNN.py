import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from scripts.ArgumentParser import ArgumentParser
from networks.AIDetection import AIDetectionCNN
from torchvision.datasets import ImageFolder

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


train_file = 'train_dataset_CNN.pt'
test_file = 'test_dataset_CNN.pt'

if os.path.exists(train_file):
    train_dataset = torch.load(train_file)
else:
    train_dataset = ImageFolder(root=args.train_data_path, transform=transforms.ToTensor())
    torch.save(train_dataset, train_file)
if os.path.exists(test_file):
    test_dataset = torch.load(test_file)
else:
    test_dataset = ImageFolder(root=args.test_data_path, transform=transforms.ToTensor())
    torch.save(test_dataset, test_file)

# Предполагаем, что у вас есть загруженные данные и созданы DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Инициализация модели
model = AIDetectionCNN(input_channels=3, output_size=2).to(args.device)  # Предполагаем, что два класса (например, настоящие и фальшивые изображения)
#criterion = nn.CrossEntropyLoss().to(args.device)
criterion = nn.BCEWithLogitsLoss().to(args.device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
model.train()
# Обучение модели
for epoch in range(args.epochs):
    running_loss = 0.0
    train_labels = []
    train_predictions = []
    print(f"Epoch [{epoch+1}/{args.epochs}]")
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(args.device), labels.to(args.device)

        optimizer.zero_grad()
        outputs = model(inputs)
        one_hot_labels = nn.functional.one_hot(labels, 2)
        loss = criterion(outputs, one_hot_labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

        outputs = torch.sigmoid(outputs)  # Преобразуем выходы в бинарные предсказания
        _, preds = torch.max(outputs, 1)
        train_labels.extend(labels.cpu().numpy())
        train_predictions.extend(preds.detach().cpu().numpy())
    epoch_loss = running_loss / len(train_loader.dataset)
    #print(f"Loss: {epoch_loss / len(train_loader)}")
    culc_confusion_matrix(train_labels, train_predictions)

    torch.save(model, "./saveModel/AIDetectionModels/" + f'modelAIDetection_epoch_{epoch+1}.pth')

    print("Test data acc")
    # Вычисление матрицы ошибок на обучающей выборке
    model.eval()
    train_preds = []
    train_targets = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)  # Преобразуем выходы в бинарные предсказания
            _, preds = torch.max(outputs, 1)
            train_labels.extend(labels.cpu().numpy())
            train_predictions.extend(preds.detach().cpu().numpy())
    culc_confusion_matrix(train_labels, train_predictions)
