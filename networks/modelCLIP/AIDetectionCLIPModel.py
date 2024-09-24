import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, ViTImageProcessor, ViTModel, DeiTModel, DeiTFeatureExtractor
import torch.nn.init as init
import torchvision.models as models
from transformers import ViTModel, ViTConfig
import torchvision.transforms as transforms
from transformers import ViTModel, ViTFeatureExtractor
from torchvision.models import resnet101

# Словарь, сопоставляющий названия моделей с классами моделей и экстракторами признаков
model_dict = {
    "openai/clip-vit-base-patch32": (CLIPModel, CLIPProcessor),
    "openai/clip-resnet-50": (CLIPModel, CLIPProcessor),
    "google/vit-base-patch16-224": (ViTModel, ViTImageProcessor),
    "google/vit-large-patch16-224": (ViTModel, ViTImageProcessor)
}
    
class Classifierclip_deit_base_patch16_224_L7(nn.Module):
    def __init__(self, output_dim=2):
        super(Classifierclip_deit_base_patch16_224_L7, self).__init__()

        self.model = DeiTModel.from_pretrained("facebook/deit-base-patch16-224")

        self.fc_layers = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )

    def forward(self, x):
        features = self.model(x).last_hidden_state
        features = features[:, 0, :] # Берем только первый токен
        output = self.fc_layers(features)
        return output

class Classifierclip_vit_base_patch32(nn.Module):
    def __init__(self, feature_dim=512, output_dim=2):
        super(Classifierclip_vit_base_patch32, self).__init__()
        self.input_dim = feature_dim
        self.output_dim = output_dim
        # Загрузка предобученной модели CLIP
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
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
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )

    def forward(self, x):
        # Прогоняем входной тензор через модель CLIP и слой классификации
        features = self.clip.get_image_features(pixel_values=x)
        x = self.classifier(features)
        return x

class ClassifierResNet50(nn.Module):
    def __init__(self, feature_dim=512, output_dim=2):
        super(ClassifierResNet, self).__init__()
        self.input_dim = feature_dim
        self.output_dim = output_dim
        # Загрузка предобученной модели CLIP
        self.clip = CLIPModel.from_pretrained("google/clip-resnet-50")
        self.processor = CLIPProcessor.from_pretrained("google/clip-resnet-50")
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
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )

    def forward(self, x):
        # Прогоняем входной тензор через модель CLIP и слой классификации
        features = self.clip.get_image_features(pixel_values=x)
        x = self.classifier(features)
        return x

class ClassifierResNet(nn.Module):
    def __init__(self, feature_dim=2048, output_dim=2):
        super(ClassifierResNet, self).__init__()
        self.input_dim = feature_dim
        self.output_dim = output_dim
        # Загрузка предобученной модели ResNet-101
        self.resnet101 = resnet101(pretrained=True)
        # Заменяем последний слой (fully connected layer) на слой для классификации
        self.resnet101.fc = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )

    def forward(self, x):
        # Прогоняем входной тензор через модель ResNet-101 и слой классификации
        x = self.resnet101(x)
        return x

class ClassifierViT(nn.Module):
    def __init__(self, feature_dim=768, output_dim=2):
        super(ClassifierViT, self).__init__()
        self.input_dim = feature_dim
        self.output_dim = output_dim
        # Загрузка предобученной модели ViT
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
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
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )

    def forward(self, x):
        # Прогоняем входной тензор через модель ViT и слой классификации
        features = self.vit(x)['last_hidden_state']
        x = self.classifier(features[:, 0, :])  # Векторизуем изображение и применяем классификатор
        return x
    
