import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_size=(3, 64, 64)):
        super(Net, self).__init__()
        input_channels, input_height, input_width = input_size
        # Сверточные слои
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        # Полносвязные слои
        self.fc1 = nn.Linear(128 * (input_height // 8) * (input_width // 8), 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)  # Для бинарной классификации
        
    def forward(self, x):
        # Проход через сверточные слои с функциями активации
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # Пулинговый слой
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # Пулинговый слой
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)  # Пулинговый слой
        # "Распрямление" данных перед подачей на полносвязные слои
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def save_weights(self, epoch, dir_path):
        # Создание директории, если её нет
        os.makedirs(dir_path, exist_ok=True)
        # Генерация имени файла
        file_name = os.path.join(dir_path, f'model_epoch_{epoch}.pth')
        # Сохранение весов модели на указанное устройство
        checkpoint = {
            'model_state_dict': self.state_dict(),
        }
        torch.save(checkpoint, file_name)
        print(f'Model weights saved to {file_name}')

    
    def load_weights(self, file_path):
        # Загрузка весов модели из указанного файла
        checkpoint = torch.load(file_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        print(f'Model weights loaded from epoch {epoch}')