import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, ViTImageProcessor, ViTModel
import torch.nn.init as init
import torchvision.models as models
from transformers import ViTModel, ViTConfig
import torchvision.transforms as transforms

# Словарь, сопоставляющий названия моделей с классами моделей и экстракторами признаков
model_dict = {
    "openai/clip-vit-base-patch32": (CLIPModel, CLIPProcessor),
    "openai/clip-resnet-50": (CLIPModel, CLIPProcessor),
    "google/vit-base-patch16-224": (ViTModel, ViTImageProcessor),
    "google/vit-large-patch16-224": (ViTModel, ViTImageProcessor)
}

class Classifierclip_vit_base_patch32_L7(nn.Module):
    def __init__(self, model_name, output_dim=2):
        super(Classifierclip_vit_base_patch32_L7, self).__init__()

        self.model_name = model_name

        self.fc_layers = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 8192),
            nn.ReLU(),
            nn.Linear(8192, 8192),
            nn.ReLU(),
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_dim)
        )

    def forward(self, x):
        x = self.fc_layers(x)
        return x

class Classifierclip_vit_base_patch32(nn.Module):
    def __init__(self, model_name, output_dim=2):
        super(Classifierclip_vit_base_patch32, self).__init__()
        self.input_dim = 512
        self.output_dim = output_dim
        # Загрузка предобученной модели CLIP
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # Заменяем последний слой (fully connected layer) на слой для классификации
        self.classifier = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        # Прогоняем входной тензор через модель CLIP и слой классификации
        features = self.clip.get_image_features(pixel_values=x)
        x = self.classifier(features)
        return x
    
