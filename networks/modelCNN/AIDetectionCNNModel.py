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
        #self.dropout = nn.Dropout(dropout_rate)

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
        #x = self.dropout(x)
        
        # Выходной слой
        x = self.fc2(x)
        
        return x
    
class AIDetectionCNN_split_Linear_layers(nn.Module):
    def __init__(self, input_channels, output_size, dropout_rate=0.2):
        super(AIDetectionCNN_split_Linear_layers, self).__init__()
        
        # Увеличиваем размеры сверточных слоев
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        
        # Адаптивный пулинг для приведения изображений к фиксированному размеру
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # Обновленный размер данных после адаптивного пулинга
        flattened_size = 256 * 8 * 8
        
        # Увеличиваем размеры полносвязных слоев
        self.fc1 = nn.Linear(flattened_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        
        # Выходной полносвязный слой
        self.fc5 = nn.Linear(128, output_size)
        
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
        x = x.view(-1, 256 * 8 * 8)
        
        # Прогоняем данные через полносвязные слои
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
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
    def __init__(self, input_channels, output_size, dropout_rate=0.5):
        super(AIDetectionCNN_split_Linear_layers_NormBatch, self).__init__()

        # Сверточные слои с Batch Normalization и активацией PReLU
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.PReLU()  # PReLU активация
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.PReLU()  # PReLU активация
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.PReLU()  # PReLU активация
        
        # Используем Global Average Pooling вместо адаптивного пулинга
        self.global_avg_pool = nn.AdaptiveAvgPool2d((2, 2))
        
        # Полносвязные слои
        flattened_size = 128 * 2 * 2  # Размер данных после Global Average Pooling
        self.fc1 = nn.Linear(flattened_size, 512)
        self.relu_fc1 = nn.PReLU()  # PReLU активация
        self.fc2 = nn.Linear(512, 1024)
        self.relu_fc2 = nn.PReLU()  # PReLU активация
        self.fc3 = nn.Linear(1024, 256)
        self.relu_fc3 = nn.PReLU()  # PReLU активация
        self.fc4 = nn.Linear(256, 128)
        self.relu_fc4 = nn.PReLU()  # PReLU активация
        self.fc5 = nn.Linear(128, output_size)
        
        # Дроп-аут для регуляризации
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Проход через сверточные слои с Batch Normalization и активацией PReLU
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        # Проход через Global Average Pooling
        x = self.global_avg_pool(x)
        
        # Преобразование к размеру (batch_size, flattened_size)
        x = x.view(x.size(0), -1)
        
        # Проход через полносвязные слои
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout(x)  # Применение Dropout
        
        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.relu_fc3(x)
        x = self.dropout(x)
        
        x = self.fc4(x)
        x = self.relu_fc4(x)
        x = self.dropout(x)
        
        # Выходной слой
        x = self.fc5(x)
        
        return x
    
class AIDetectionCNN7(nn.Module):
    def __init__(self, input_channels, output_size, dropout_rate=0.2):
        super(AIDetectionCNN7, self).__init__()
        
        # Определяем сверточные слои с нарастающим количеством фильтров
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)

        # Используем пулинг после каждого сверточного слоя
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Адаптивный пулинг для приведения изображений к фиксированному размеру
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        # Обновленный размер данных после адаптивного пулинга
        flattened_size = 128 * 8 * 8
        
        # Определяем полносвязные слои
        self.fc1 = nn.Linear(flattened_size, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 2048)
        
        # Выходной слой
        self.fc4 = nn.Linear(2048, output_size)
        
        # Дроп-аут для регуляризации
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Прогоняем входные данные через сверточные слои
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = F.relu(self.conv7(x))
        x = self.pool(x)
        
        # Адаптивный пулинг
        x = self.adaptive_pool(x)
        
        # Разглаживаем данные перед полносвязными слоями
        x = x.view(-1, 128 * 8 * 8)
        
        # Прогоняем данные через полносвязные слои
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        
        # Выходной слой
        x = self.fc4(x)
        
        return x

class AIDetectionCNN7BatchNorm(nn.Module):
    def __init__(self, input_channels, output_size, dropout_rate=0.2):
        super(AIDetectionCNN7BatchNorm, self).__init__()

        # Определяем сверточные слои с нарастающим количеством фильтров
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(128)

        # Макс-пулинг после каждого сверточного слоя
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Адаптивный пулинг для приведения изображений к фиксированному размеру
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        # Размер после адаптивного пулинга
        flattened_size = 128 * 8 * 8

        # Полносвязные слои с регуляризацией и нормализацией
        self.fc1 = nn.Linear(flattened_size, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.bn_fc2 = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, 2048)
        self.bn_fc3 = nn.BatchNorm1d(2048)
        
        # Выходной слой
        self.fc4 = nn.Linear(2048, output_size)

        # Дроп-аут для регуляризации
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Прохождение через сверточные слои с Batch Normalization и Leaky ReLU
        x = nn.LeakyReLU()(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = nn.LeakyReLU()(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = nn.LeakyReLU()(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = nn.LeakyReLU()(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = nn.LeakyReLU()(self.bn5(self.conv5(x)))
        x = self.pool(x)
        x = nn.LeakyReLU()(self.bn6(self.conv6(x)))
        x = self.pool(x)
        x = nn.LeakyReLU()(self.bn7(self.conv7(x)))
        x = self.pool(x)

        # Адаптивный пулинг для приведения размера к фиксированному значению
        x = self.adaptive_pool(x)

        # Разглаживание данных перед отправкой в полносвязные слои
        x = x.view(-1, 128 * 8 * 8)

        # Прогон через полносвязные слои с Batch Normalization, дроп-аутом и Leaky ReLU
        x = nn.LeakyReLU()(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = nn.LeakyReLU()(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = nn.LeakyReLU()(self.bn_fc3(self.fc3(x)))

        # Выходной слой
        x = self.fc4(x)

        return x