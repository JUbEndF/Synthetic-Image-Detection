import torch
from torch import nn
from networks.ClassifierBasedNoiseSignal.DenoisingModel import NoiseExtractionNetwork

# Создание шумоподавляющей сети (пример с DnCNN)
class DenoisingNetwork(nn.Module):
    def __init__(self):
        super(DenoisingNetwork, self).__init__()
        # Пример DnCNN для шумоподавления
        self.repo_url = "./saveModel/DenoisingNetwork/DenoisingNetwork_epoch_11_Lost_0.0003881548993869203.pth"
        self.model = torch.load(self.repo_url)
        self.model.eval()

    def forward(self, x):
        # Обработка входного изображения
        with torch.no_grad():
            return self.model(x)


class NoiseExtractionClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_rate=0.2):
        """
        Создает сеть, которая объединяет сеть для выделения шума из изображения и классификатор.

        :param in_channels: Количество входных каналов в изображении (обычно 3 для RGB-изображений).
        :param num_classes: Количество классов для классификации.
        :param num_rrg_blocks: Количество блоков `ResidualRefinementBlock` в сети для выделения шума.
        :param dropout_rate: Уровень дроп-аута для регуляризации классификатора.
        """
        super(NoiseExtractionClassifier, self).__init__()

        # Классификатор
        # Используем глобальное усреднительное пулирование, чтобы привести данные к фиксированному размеру
        self.global_pool = nn.AdaptiveAvgPool2d((32, 32))
        
        # Полносвязные слои
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_channels * 32 * 32, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.fc4 = nn.Linear(128, 64)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(dropout_rate)

        self.fc5 = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        Прогоняет входные изображения через сеть для выделения шума, затем через классификатор.

        :param x: Входные изображения (torch.Tensor).
        :return: Предсказания (torch.Tensor).
        """
        
        # Глобальное усреднительное пулирование для приведения данных к фиксированному размеру
        noise = self.global_pool(x)
        noise = self.global_pool(x)
        
        # Классификация выделенного шума
        x = self.flatten(noise)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.dropout4(x)

        predictions = self.fc5(x)
        
        return predictions
    
class NoiseExtractionClassifierImproved(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_rate=0.3, l2_reg=0.01):
        super(NoiseExtractionClassifierImproved, self).__init__()

        # Глобальное усреднительное пулирование
        self.global_pool = nn.AdaptiveAvgPool2d((32, 32))

        # Полносвязные слои
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_channels * 32 * 32, 1024)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(1024, 512)
        self.gelu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(512, 256)
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.1)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.fc4 = nn.Linear(256, 128)
        self.gelu4 = nn.GELU()
        self.dropout4 = nn.Dropout(dropout_rate)

        # Выходной слой
        self.fc_out = nn.Linear(128, num_classes)
        
        # Инициализация весов с L2-регуляризацией
        self._initialize_weights(l2_reg)

    def forward(self, x):
        # Глобальное усреднительное пулирование
        noise = self.global_pool(x)
        
        # Полносвязные слои
        x = self.flatten(noise)
        x = self.fc1(x)
        x = self.leaky_relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.gelu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.leaky_relu3(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        x = self.gelu4(x)
        x = self.dropout4(x)

        # Выходной слой
        predictions = self.fc_out(x)
        
        return predictions

class NoiseExtractionClassifier3LBatchNorm1d(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
        Создает сеть для классификации шумных изображений.

        :param in_channels: Количество входных каналов в изображении (обычно 3 для RGB-изображений).
        :param num_classes: Количество классов для классификации.
        """
        super(NoiseExtractionClassifier3LBatchNorm1d, self).__init__()

        # Глобальное усреднительное пулирование
        self.global_pool = nn.AdaptiveAvgPool2d((32, 32))
        
        # Классификатор
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_channels * 32 * 32, 256)
        self.norm1 = nn.BatchNorm1d(256)
        self.leaky_relu1 = nn.LeakyReLU()

        self.fc2 = nn.Linear(256, 128)
        self.norm2 = nn.BatchNorm1d(128)
        self.leaky_relu2 = nn.LeakyReLU()
        
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Прогоняет входные изображения через сеть для выделения шума, затем через классификатор.

        :param x: Входные изображения (torch.Tensor).
        :return: Предсказания (torch.Tensor).
        """
        # Приведение данных к фиксированному размеру
        noise = self.global_pool(x)
        
        # Классификация выделенного шума
        x = self.flatten(noise)
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.leaky_relu1(x)
        
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.leaky_relu2(x)
        
        predictions = self.fc3(x)
        
        return predictions