import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, ViTImageProcessor, ViTModel
import torch.nn.init as init

# Словарь, сопоставляющий названия моделей с классами моделей и экстракторами признаков
model_dict = {
    "openai/clip-vit-base-patch32": (CLIPModel, CLIPProcessor),
    "openai/clip-resnet-50": (CLIPModel, CLIPProcessor),
    "google/vit-base-patch16-224": (ViTModel, ViTImageProcessor),
    "google/vit-large-patch16-224": (ViTModel, ViTImageProcessor)
}

class DeepClassifierCLIP16384(nn.Module):
    def __init__(self, model_name, feature_dim=512, output_dim=2, dropout_rate=0):
        super(DeepClassifierCLIP16384, self).__init__()

        self.model_name = model_name

        self.fc_layers = nn.Sequential(
            nn.Linear(feature_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 8192),
            nn.ReLU(inplace=True),
            nn.Linear(8192, 16384),
            nn.ReLU(inplace=True),
            nn.Linear(16384, 8192),
            nn.ReLU(inplace=True),
            nn.Linear(8192, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, output_dim)
        )

    def forward(self, x):
        x = self.fc_layers(x)
        return x

class DeepCLIPClassifier(nn.Module):
    def __init__(self, model_name, feature_dim, hidden_dim=4096, num_layers=50, output_dim=2, dropout_rate=0.1):
        super(DeepCLIPClassifier, self).__init__()
        self.model_name = model_name
        # Архитектура классификатора
        layers = []
        layers.append(nn.Linear(feature_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        # Добавляем скрытые слои
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        # Выходной слой
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        # Пропускаем через полносвязные слои
        x = self.fc_layers(x)
        return x


class ImprovedClassifierCLIP12LBatchNorm1d(nn.Module):
    def __init__(self, model_name, feature_dim=512, output_dim=2, dropout_rate=0):
        super(ImprovedClassifierCLIP12LBatchNorm1d, self).__init__()

        self.model_name = model_name
        # Архитектура классификатора
        self.fc_layers = nn.Sequential(
            nn.Linear(feature_dim, 8192),
            nn.BatchNorm1d(8192),  # Добавляем нормализацию
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(8192, 4096),
            nn.BatchNorm1d(4096),  # Добавляем нормализацию
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),  # Добавляем нормализацию
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),  # Добавляем нормализацию
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(8, output_dim)
        )
        
    def forward(self, x):
        # Прогон через полносвязные слои классификатора
        x = self.fc_layers(x)
        return x
    
class ImprovedClassifierCLIP40L(nn.Module):
    def __init__(self, model_name, feature_dim=512, output_dim=2, dropout_rate=0):
        super(ImprovedClassifierCLIP40L, self).__init__()

        self.model_name = model_name
        # Архитектура классификатора
        self.fc_layers = nn.Sequential(
            nn.Linear(feature_dim, 8192),
            nn.BatchNorm1d(8192),  # Добавляем нормализацию
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(8192, 4096),
            nn.BatchNorm1d(4096),  # Добавляем нормализацию
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),  # Добавляем нормализацию
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),  # Добавляем нормализацию
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),  # Добавляем нормализацию
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),  # Добавляем нормализацию
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),  # Добавляем нормализацию
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),  # Добавляем нормализацию
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),  # Добавляем нормализацию
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),  # Добавляем нормализацию
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),  # Добавляем нормализацию
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Добавляем еще 20 слоев
            nn.Linear(8, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(4, 2),
            nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(2, 2),
            nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(2, 2),
            nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(2, 2),
            nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(2, 2),
            nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(2, 2),
            nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(2, 2),
            nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(2, 2),
            nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(2, 2),
            nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(2, 2),
            nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(2, 2),
            nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(2, 2),
            nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(2, 2),
            nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(2, 2),
            nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(2, 2),
            nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(2, 2),
            nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(2, 2),
            nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(2, 2),
            nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(2, output_dim)  # Выходной слой
        )
        
    def forward(self, x):
        # Прогон через полносвязные слои классификатора
        x = self.fc_layers(x)
        return x


class ImprovedClassifierCLIP12L(nn.Module):
    def __init__(self, model_name, feature_dim=512, output_dim=2, dropout_rate=0):
        super(ImprovedClassifierCLIP12L, self).__init__()

        self.model_name = model_name
        # Архитектура классификатора
        self.fc_layers = nn.Sequential(
            nn.Linear(feature_dim, 8192),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(8, output_dim)
        )
        
    def forward(self, x):
        # Прогон через полносвязные слои классификатора
        x = self.fc_layers(x)
        return x
    
class ImprovedClassifierCLIP12Lkaiming_normal(nn.Module):
    def __init__(self, model_name, feature_dim=512, output_dim=2, dropout_rate=0):
        super(ImprovedClassifierCLIP12Lkaiming_normal, self).__init__()

        self.model_name = model_name
        # Архитектура классификатора
        self.fc_layers = nn.Sequential(
            nn.Linear(feature_dim, 8192),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(8, output_dim)
        )
        
        # Применение инициализации весов методом He
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x):
        # Прогон через полносвязные слои классификатора
        x = self.fc_layers(x)
        return x