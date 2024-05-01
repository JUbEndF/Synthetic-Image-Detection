import torch
import torch.nn as nn
import torch.nn.functional as F


#model #1
class AIDetectionCNN(nn.Module):
    def __init__(self, input_channels, output_size, dropout_rate=0.2):
        super(AIDetectionCNN, self).__init__()

        # Увеличенные размеры каналов сверточных слоев
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        
        # Дополнительный адаптивный пулинг для уменьшения размеров данных
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # Полносвязные слои с увеличенным количеством нейронов
        self.fc1 = nn.Linear(256 * 8 * 8, 2048)  # Увеличено количество нейронов
        self.fc2 = nn.Linear(2048, output_size)  # Увеличено количество нейронов в последнем слое
        
        # Регуляризация
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Применение слоев и функций активации
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = F.relu(self.conv3(x))
        
        # Адаптивный пулинг
        x = self.adaptive_pool(x)
        
        # Разглаживание данных перед полносвязными слоями
        x = x.view(-1, 256 * 8 * 8)
        
        # Полносвязные слои с функциями активации и регуляризацией
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Выходной слой
        x = self.fc2(x)
        
        return x
    
class AIDetectionCNN_split_Linear_layers(nn.Module):
    def __init__(self, input_channels, output_size, dropout_rate=0.2):
        super(AIDetectionCNN_split_Linear_layers, self).__init__()
        
        # Сверточные слои
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Адаптивный пулинг для приведения изображений к фиксированному размеру
        self.adaptive_pool = nn.AdaptiveAvgPool2d((64, 64))
        
        # Разглаживаем данные перед подачей в полносвязные слои
        flattened_size = 64 * 64 * 64
        self.fc1 = nn.Linear(flattened_size, 32)
        
        # Добавляем несколько полносвязных слоев, каждый с 32 нейронами
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 32)
        
        # Выходной полносвязный слой
        self.fc5 = nn.Linear(32, output_size)
        
        # Дроп-аут для регуляризации
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Прогоняем входные данные через сверточные слои
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        
        # Разглаживаем данные перед подачей в полносвязные слои
        x = x.view(-1, 64 * 64 * 64)
        
        # Прогоняем данные через полносвязные слои
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        
        # Выходной полносвязный слой
        x = self.fc5(x)
        
        return x
    

class AIDetectionCNNBaseNormBatch(nn.Module):
    def __init__(self, input_channels, output_size, dropout_rate=0.2):
        super(AIDetectionCNNBaseNormBatch, self).__init__()
        # Сверточные слои с Batch Normalization и активацией Leaky ReLU
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Используем Global Average Pooling вместо адаптивного пулинга
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Полносвязные слои
        flattened_size = 64 * 1 * 1  # Размер данных после Global Average Pooling
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, output_size)
        
        # Дроп-аут для регуляризации
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Проходим через сверточные слои с активацией Leaky ReLU
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        
        # Применяем Global Average Pooling
        x = self.global_avg_pool(x)
        
        # Разглаживаем данные перед подачей в полносвязные слои
        x = x.view(-1, 64)
        
        # Проходим через полносвязные слои
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class AIDetectionCNN_split_Linear_layers_NormBatch(nn.Module):
    def __init__(self, input_channels, output_size, dropout_rate=0.2):
        super(AIDetectionCNN_split_Linear_layers_NormBatch, self).__init__()
        # Сверточные слои с Batch Normalization и активацией PReLU
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Используем Global Average Pooling вместо адаптивного пулинга
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Полносвязные слои
        flattened_size = 64 * 1 * 1  # Размер данных после Global Average Pooling
        # Разделяем полносвязные слои
        self.fc1 = nn.Linear(flattened_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 32)
        self.fc5 = nn.Linear(32, output_size)
        
        # Дроп-аут для регуляризации
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Проходим через сверточные слои с активацией PReLU
        x = F.prelu(self.conv1(x), torch.ones(1).to(x.device))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.prelu(self.conv2(x), torch.ones(1).to(x.device))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.prelu(self.conv3(x), torch.ones(1).to(x.device))
        
        # Применяем Global Average Pooling
        x = self.global_avg_pool(x)
        
        # Разглаживаем данные перед подачей в полносвязные слои
        x = x.view(-1, 64)
        
        # Проходим через разделенные полносвязные слои с активацией PReLU
        x = F.prelu(self.fc1(x), torch.ones(1).to(x.device))
        x = self.dropout(x)
        x = F.prelu(self.fc2(x), torch.ones(1).to(x.device))
        x = self.dropout(x)
        x = F.prelu(self.fc3(x), torch.ones(1).to(x.device))
        x = self.dropout(x)
        x = F.prelu(self.fc4(x), torch.ones(1).to(x.device))
        x = self.dropout(x)
        
        # Выходной полносвязный слой
        x = self.fc5(x)
        
        return x