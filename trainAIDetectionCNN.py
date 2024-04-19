import os
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import auc, confusion_matrix, roc_auc_score, roc_curve
from tqdm import tqdm
from scripts.ArgumentParser import ArgumentParser
from networks.AIDetection import AIDetectionCNN, AIDetectionCNNsave
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
    return cm

train_file = './data/MyDataset/train_dataset_CNN_1024.pt'
test_file = './data/MyDataset/test_dataset_CNN_1024.pt'

if os.path.exists(train_file):
    train_dataset = torch.load(train_file)
else:
    train_dataset = ImageFolder(root=args.train_data_path, transform=transforms.ToTensor())
    #torch.save(train_dataset, train_file)
if os.path.exists(test_file):
    test_dataset = torch.load(test_file)
else:
    test_dataset = ImageFolder(root=args.val_data_path, transform=transforms.ToTensor())
    #torch.save(test_dataset, test_file)

# Предполагаем, что у вас есть загруженные данные и созданы DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Инициализация модели

model = AIDetectionCNNsave(input_channels=3, output_size=2).to(args.device)  # Предполагаем, что два класса (например, настоящие и фальшивые изображения)

AUCROC_train = []
AUCROC_test = []

#criterion = nn.CrossEntropyLoss().to(args.device)
criterion = nn.BCEWithLogitsLoss().to(args.device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Обучение модели
for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0
    train_labels = []
    predictions_probs = []
    train_predictions = []
    print(f"Epoch [{epoch+1}/{args.epochs}]")
    for inputs, labels in tqdm(train_loader):
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

        predictions_probs.extend(outputs.detach().cpu().numpy()[:, 1])
        train_labels.extend(labels.cpu().numpy())
        train_predictions.extend(preds.detach().cpu().numpy())
    epoch_loss = running_loss / len(train_loader.dataset)
    #print(f"Loss: {epoch_loss / len(train_loader)}")
    culc_confusion_matrix(train_labels, train_predictions)

    fpr, tpr, _ = roc_curve(train_labels, predictions_probs)
    auc = roc_auc_score(train_labels, predictions_probs)
    AUCROC_train.append([epoch, fpr, tpr, auc])
    #create ROC curve
    

    torch.save(model, "./saveModel/AIDetectionModels/" + f'modelAIDetection_epoch_{epoch+1}.pth')

    print("Test data acc")
    # Вычисление матрицы ошибок на обучающей выборке
    model.eval()
    train_predictions = []
    train_labels = []
    predictions_probs = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)  # Преобразуем выходы в бинарные предсказания
            _, preds = torch.max(outputs, 1)
            predictions_probs.extend(outputs.detach().cpu().numpy()[:, 1])
            train_labels.extend(labels.cpu().numpy())
            train_predictions.extend(preds.detach().cpu().numpy())
    culc_confusion_matrix(train_labels, train_predictions)

    fpr, tpr, _ = roc_curve(train_labels, predictions_probs)
    auc = roc_auc_score(train_labels, predictions_probs)
    AUCROC_test.append([epoch, fpr, tpr, auc])

# Построение ROC-кривых
for data in AUCROC_train:
    epoch, fpr, tpr, auc = data
    plt.plot(fpr, tpr, label=f'Эпоха {epoch} (AUC = {auc:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Случайный классификатор')
# Дополнительные настройки графика
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Train')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Построение ROC-кривых
for data in AUCROC_test:
    epoch, fpr, tpr, auc = data
    plt.plot(fpr, tpr, label=f'Эпоха {epoch} (AUC = {auc:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Случайный классификатор')
# Дополнительные настройки графика
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Test')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()