import torch
import torch.nn as nn
import torch.nn.functional as F

class AIDetectionClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(AIDetectionClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc5 = nn.Linear(hidden_sizes[3], hidden_sizes[4])
        self.fc6 = nn.Linear(hidden_sizes[4], num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = torch.sigmoid(self.fc6(x))
        return x
    
#провал не использовать
class AIDetectionClassifierLeakyReLU(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(AIDetectionClassifierLeakyReLU, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc5 = nn.Linear(hidden_sizes[3], hidden_sizes[4])
        self.fc6 = nn.Linear(hidden_sizes[4], 1)
        self.leaky_relu = nn.LeakyReLU(0.1)  # Устанавливаем небольшой коэффициент наклона alpha=0.1
    
    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.leaky_relu(self.fc4(x))
        x = self.leaky_relu(self.fc5(x))
        x = torch.sigmoid(self.fc6(x))
        return x
    

class AIDetectionCNN(nn.Module):
    def __init__(self, input_channels, output_size, dropout_rate=0.2):
        super(AIDetectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(512 * 8 * 8, 1024)  # 8*8*512 - размер данных после пятого сверточного слоя
        self.fc2 = nn.Linear(1024, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))  # Адаптивный пулинг для приведения изображений к фиксированному размеру

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv5(x))
        x = self.adaptive_pool(x)
        x = x.view(-1, 512 * 8 * 8)  # Разглаживаем данные перед подачей в полносвязные слои
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class AIDetectionCNNsave(nn.Module):
    def __init__(self, input_channels, output_size, dropout_rate=0.2):
        super(AIDetectionCNNsave, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 64 * 64, 128)  # 64*64*64 - размер данных после третьего сверточного слоя
        self.fc2 = nn.Linear(128, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((64, 64))  # Адаптивный пулинг для приведения изображений к фиксированному размеру

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = x.view(-1, 64 * 64 * 64)  # Разглаживаем данные перед подачей в полносвязные слои
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class AIDetectionCNNGP(nn.Module):
    def __init__(self, input_channels, output_size, dropout_rate=0.2):
        super(AIDetectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Убираем указание фиксированного размера для полносвязного слоя
        self.fc1 = nn.Linear(64 * 64 * 64, 128)
        
        self.fc2 = nn.Linear(128, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Убираем адаптивный пулинг
        # self.adaptive_pool = nn.AdaptiveAvgPool2d((64, 64))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        
        # Используем глобальное пулинг для агрегации признаков
        x = F.avg_pool2d(x, kernel_size=x.size()[2:])

        x = x.view(-1, 64 * 64 * 64)  # Разглаживаем данные перед подачей в полносвязные слои
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x