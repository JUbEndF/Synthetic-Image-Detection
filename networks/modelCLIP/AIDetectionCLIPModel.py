import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

def extract_clip_features(images, clip_model, clip_processor, device):
    """
    Извлекает вектор признаков из входных изображений с использованием CLIP.

    :param images: Входные изображения (torch.Tensor).
    :param clip_model: Предобученная модель CLIP.
    :param clip_processor: Процессор CLIP для подготовки изображений.
    :param device: Устройство, на котором выполняется модель (например, 'cpu' или 'cuda').
    :return: Вектор признаков, извлеченный с использованием CLIP (torch.Tensor).
    """
    # Подготовка изображений с помощью процессора CLIP
    inputs = clip_processor(images=images, return_tensors='pt')
    
    # Перемещение входных данных на нужное устройство
    for key in inputs:
        inputs[key] = inputs[key].to(device)
    
    # Извлекаем вектор признаков из входных изображений с помощью модели CLIP
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    
    return image_features


class CLIPClassifier(nn.Module):
    def __init__(self, model_name, device, output_dim=2, dropout_rate=0.3, l2_reg=0.01):
        """
        Создает классификатор, который принимает вектор признаков от CLIP.

        :param model_name: Название предобученной модели CLIP ("openai/clip-vit-base-patch32" или "openai/clip-resnet-50").
        :param device: Устройство для размещения модели (CPU или GPU).
        :param output_dim: Размерность выходного слоя (количество классов).
        :param dropout_rate: Уровень дроп-аута для регуляризации.
        :param l2_reg: Уровень L2 регуляризации для предотвращения переобучения.
        """
        super(CLIPClassifier, self).__init__()
        
        # Загружаем предобученную модель CLIP и процессор для подготовки входных данных
        self.clip_model = CLIPModel.from_pretrained(model_name).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)
        
        # Размер вектора признаков, извлекаемых CLIP
        clip_output_dim = 512 if model_name == "openai/clip-vit-base-patch32" else 1024
        
        # Полносвязные слои классификатора с большим количеством нейронов
        self.fc1 = nn.Linear(clip_output_dim, 512)
        self.fc1_bn = nn.BatchNorm1d(512)  # Нормализация слоя
        self.fc2 = nn.Linear(512, 256)
        self.fc2_bn = nn.BatchNorm1d(256)  # Нормализация слоя
        self.fc3 = nn.Linear(256, 128)
        self.fc3_bn = nn.BatchNorm1d(128)  # Нормализация слоя
        self.fc4 = nn.Linear(128, 64)
        self.fc4_bn = nn.BatchNorm1d(64)  # Нормализация слоя
        self.fc5 = nn.Linear(64, output_dim)
        
        # Дроп-аут для регуляризации
        self.dropout = nn.Dropout(dropout_rate)
        
        # L2 регуляризация (weight decay)
        self.l2_reg = l2_reg
        
        # Функция активации Swish (SiLU)
        self.activation = nn.SiLU()  # Swish
        
    def forward(self, x):
        """
        Прогоняет входные изображения через CLIP, затем через полносвязные слои классификатора.

        :param images: Входные изображения (torch.Tensor).
        :return: Выходные предсказания (torch.Tensor).
        """

        # Полносвязные слои с активацией Swish (SiLU), дроп-аутом и нормализацией слоя
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc1_bn(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc2_bn(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.activation(x)
        x = self.fc3_bn(x)
        x = self.dropout(x)
        
        x = self.fc4(x)
        x = self.activation(x)
        x = self.fc4_bn(x)
        x = self.dropout(x)
        
        # Выходной слой
        x = self.fc5(x)
        
        return x
    
class CLIPClassifierIncreasedNumberLayers(nn.Module):
    def __init__(self, model_name, device, output_dim=2, dropout_rate=0.2):
        """
        Создает классификатор, который принимает вектор признаков от CLIP.

        :param model_name: Название предобученной модели CLIP ("openai/clip-vit-base-patch32" или "openai/clip-resnet-50").
        :param output_dim: Размерность выходного слоя (количество классов).
        :param dropout_rate: Уровень дроп-аута для регуляризации.
        """
        super(CLIPClassifierIncreasedNumberLayers, self).__init__()
        
        # Загружаем предобученную модель CLIP и процессор для подготовки входных данных
        self.clip_model = CLIPModel.from_pretrained(model_name).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)
        
        # Размер вектора признаков, извлекаемых CLIP
        clip_output_dim = 512 if model_name == "openai/clip-vit-base-patch32" else 1024
        
        # Полносвязные слои классификатора с увеличенным количеством слоев
        self.fc1 = nn.Linear(clip_output_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, output_dim)
        
        # Дроп-аут для регуляризации
        self.dropout = nn.Dropout(dropout_rate)
        
        # Функция активации
        self.activation = nn.ReLU()
        
    def forward(self, images):
        """
        Прогоняет входные изображения через CLIP, затем через полносвязные слои классификатора.

        :param images: Входные изображения (torch.Tensor).
        :return: Выходные предсказания (torch.Tensor).
        """
        
        # Прогоняем вектор признаков через полносвязные слои классификатора
        x = self.activation(self.fc1(images))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.activation(self.fc3(x))
        x = self.dropout(x)
        x = self.activation(self.fc4(x))
        x = self.dropout(x)
        x = self.activation(self.fc5(x))
        x = self.dropout(x)
        
        # Выходной слой
        x = self.fc6(x)
        
        return x

class CLIPClassifierCrushLayers(nn.Module):
    def __init__(self, model_name, device, output_dim=2, dropout_rate=0.2):
        """
        Создает классификатор, который принимает вектор признаков от CLIP и использует больше полносвязных слоев, сохраняя размеры `512`, `256`.

        :param model_name: Название предобученной модели CLIP ("openai/clip-vit-base-patch32" или "openai/clip-resnet-50").
        :param output_dim: Размерность выходного слоя (количество классов).
        :param dropout_rate: Уровень дроп-аута для регуляризации.
        """
        super(CLIPClassifierCrushLayers, self).__init__()
        
        # Загружаем предобученную модель CLIP
        self.clip_model = CLIPModel.from_pretrained(model_name).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)
        
        # Размер вектора признаков, извлекаемых CLIP
        clip_output_dim = 512 if model_name == "openai/clip-vit-base-patch32" else 1024
        
        # Дробление слоя fc1 на 4 слоя с увеличенными размерами
        self.fc1_1 = nn.Linear(clip_output_dim, 128)  # Увеличено с 64 до 128
        self.fc1_2 = nn.Linear(128, 128)  # Увеличено с 64 до 128
        self.fc1_3 = nn.Linear(128, 128)  # Увеличено с 64 до 128
        self.fc1_4 = nn.Linear(128, 128)  # Увеличено с 64 до 128
        
        # Дробление слоя fc2 на 4 слоя с увеличенными размерами
        self.fc2_1 = nn.Linear(128, 64)  # Увеличено с 32 до 64
        self.fc2_2 = nn.Linear(64, 64)  # Увеличено с 32 до 64
        self.fc2_3 = nn.Linear(64, 64)  # Увеличено с 32 до 64
        self.fc2_4 = nn.Linear(64, 64)  # Увеличено с 32 до 64
        
        # Выходной слой
        self.fc_out = nn.Linear(64, output_dim)
        
        # Дроп-аут для регуляризации
        self.dropout = nn.Dropout(dropout_rate)
        
        # Функция активации
        self.activation = nn.ReLU()
        
    def forward(self, x):
        """
        Прогоняет входные изображения через CLIP, затем через полносвязные слои классификатора.

        :param x: Входные изображения (torch.Tensor).
        :return: Выходные предсказания (torch.Tensor).
        """
        # Прогоняем вектор признаков через дробленные слои fc1
        x = self.activation(self.fc1_1(x))
        x = self.dropout(x)
        x = self.activation(self.fc1_2(x))
        x = self.dropout(x)
        x = self.activation(self.fc1_3(x))
        x = self.dropout(x)
        x = self.activation(self.fc1_4(x))
        x = self.dropout(x)
        
        # Прогоняем через дробленные слои fc2
        x = self.activation(self.fc2_1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2_2(x))
        x = self.dropout(x)
        x = self.activation(self.fc2_3(x))
        x = self.dropout(x)
        x = self.activation(self.fc2_4(x))
        x = self.dropout(x)
        
        # Выходной слой
        x = self.fc_out(x)
        
        return x