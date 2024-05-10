import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, ViTImageProcessor, ViTModel

# Словарь, сопоставляющий названия моделей с классами моделей и экстракторами признаков
model_dict = {
    "openai/clip-vit-base-patch32": (CLIPModel, CLIPProcessor),
    "openai/clip-resnet-50": (CLIPModel, CLIPProcessor),
    "google/vit-base-patch16-224": (ViTModel, ViTImageProcessor),
    "google/vit-large-patch16-224": (ViTModel, ViTImageProcessor)
}

class ImprovedClassifierCLIP(nn.Module):
    def __init__(self, model_name, feature_dim=512, output_dim=2, dropout_rate=0.2):
        super(ImprovedClassifierCLIP, self).__init__()

        self.model_name = model_name
        # Архитектура классификатора
        self.fc_layers = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, output_dim)
        )
        
        # Активационная функция
        self.activation = nn.ReLU()

    def forward(self, x):

        # Прогон через полносвязные слои классификатора
        x = self.fc_layers(x)
        return x


class GenericClassifier(nn.Module):
    def __init__(self, feature_dim, device="cuda", output_dim=2, dropout_rate=0.05):
        """
        Создает классификатор, который принимает вектор признаков от указанной модели (ViT или CLIP).

        :param model_name: Название модели, например "vit" или "clip".
        :param device: Устройство для размещения модели (CPU или GPU).
        :param output_dim: Размерность выходного слоя (количество классов).
        :param dropout_rate: Уровень дроп-аута для регуляризации.
        """
        super(GenericClassifier, self).__init__()
        
        # Полносвязные слои классификатора
        self.fc1 = nn.Linear(feature_dim, 256)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.fc2_bn = nn.BatchNorm1d(64)
        self.fc_out = nn.Linear(64, output_dim)
        
        # Дроп-аут для регуляризации
        self.dropout = nn.Dropout(dropout_rate)
        
        # Функция активации
        self.activation = nn.ReLU()
    
    def forward(self, x):
        """
        Прогоняет входные изображения через модель, затем через полносвязные слои классификатора.

        :param x: Входные изображения (torch.Tensor).
        :return: Выходные предсказания (torch.Tensor).
        """
        
        # Прогон через полносвязные слои классификатора
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc1_bn(x)
        
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc2_bn(x)
        
        # Выходной слой
        x = self.fc_out(x)
        
        return x

class GenericClassifierCrushLayers(nn.Module):
    def __init__(self, feature_dim=512, device="cuda", output_dim=2, dropout_rate=0.05):
        """
        Создает классификатор, который принимает вектор признаков от CLIP.

        :param model_name: Название предобученной модели CLIP ("openai/clip-vit-base-patch32" или "openai/clip-resnet-50").
        :param output_dim: Размерность выходного слоя (количество классов).
        :param dropout_rate: Уровень дроп-аута для регуляризации.
        """
        super(GenericClassifierCrushLayers, self).__init__()
        
        # Полносвязные слои классификатора без дробления на подслои
        self.fc1 = nn.Linear(feature_dim, 512)
        self.fc1_bn = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, 256)
        self.fc2_bn = nn.BatchNorm1d(256)
        
        # Выходной слой
        self.fc_out = nn.Linear(256, output_dim)
        
        # Дроп-аут для регуляризации
        self.dropout = nn.Dropout(dropout_rate)
        
        # Функция активации
        self.activation = nn.LeakyReLU()  # Попробуйте Leaky ReLU

    def forward(self, x):
        """
        Прогоняет входные изображения через CLIP, затем через полносвязные слои классификатора.

        :param x: Входные изображения (torch.Tensor).
        :return: Выходные предсказания (torch.Tensor).
        """
        
        # Прогон через полносвязной слой fc1
        x = self.activation(self.fc1(x))
        x = self.fc1_bn(x)
        x = self.dropout(x)
        
        # Прогон через полносвязной слой fc2
        x = self.activation(self.fc2(x))
        x = self.fc2_bn(x)
        x = self.dropout(x)
        
        # Выходной слой
        x = self.fc_out(x)
        return x