import torch
from torch import nn
from networks.ClassifierBasedNoiseSignal.DenoisingModel import NoiseExtractionNetwork

class NoiseExtractionClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, num_rrg_blocks=4, dropout_rate=0.2):
        """
        Создает сеть, которая объединяет сеть для выделения шума из изображения и классификатор.

        :param in_channels: Количество входных каналов в изображении (обычно 3 для RGB-изображений).
        :param num_classes: Количество классов для классификации.
        :param num_rrg_blocks: Количество блоков `ResidualRefinementBlock` в сети для выделения шума.
        :param dropout_rate: Уровень дроп-аута для регуляризации классификатора.
        """
        super(NoiseExtractionClassifier, self).__init__()

        # Создание сети для выделения шума из изображения
        self.noise_extraction_network = NoiseExtractionNetwork(in_channels, num_rrg_blocks)

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
        # Выделение шума из входного изображения
        noise = self.noise_extraction_network(x)
        
        # Глобальное усреднительное пулирование для приведения данных к фиксированному размеру
        noise = self.global_pool(noise)
        
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