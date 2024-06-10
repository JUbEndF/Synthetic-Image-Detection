import torch
from torch import nn
from transformers import ViTModel, ViTFeatureExtractor
from torchviz import make_dot
import torchvision.transforms as transforms
from PIL import Image

class ClassifierViT(nn.Module):
    def __init__(self, feature_dim=768, output_dim=2):
        super(ClassifierViT, self).__init__()
        self.input_dim = feature_dim
        self.output_dim = output_dim
        # Заменяем последний слой (fully connected layer) на слой для классификации
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )

    def forward(self, x):
        x = self.classifier(x)  # Векторизуем изображение и применяем классификатор
        return x

if __name__ == '__main__':
    model = ClassifierViT()
    # Создаем пример входных данных
    example_input = torch.randn(3, 768)  

    # Пропустите тестовый тензор через модель
    output_tensor = model(example_input)

    # Визуализируйте модель
    dot = make_dot(output_tensor, params=dict(model.named_parameters()))
    dot.node_attr.update(fontsize='50')

    # Группировка узлов
    for i, layer in enumerate(model.classifier):
        if isinstance(layer, nn.Linear):
            #  Изменяем метку узла для линейных слоев
            dot.node_attr['label'] = f'Linear {i}'
        elif isinstance(layer, nn.BatchNorm1d):
            #  Изменяем метку узла для BatchNorm1d
            dot.node_attr['label'] = f'BatchNorm1d {i}'
        elif isinstance(layer, nn.ReLU):
            #  Изменяем метку узла для ReLU
            dot.node_attr['label'] = f'ReLU {i}'

    dot.render("AIDetectionCLIP", format="png")
    """
    # Создаем пример входных данных
    example_input = torch.randn(3,768)  # Замените на ваши входные данные

   # Пропустите тестовый тензор через модель
    output_tensor = model(example_input)
    
    # Визуализируйте модель
    dot = make_dot(output_tensor, params=dict(model.named_parameters()))
    dot.node_attr.update(fontsize='50')
    dot.render("AIDetectionCLIP", format="png")
    """