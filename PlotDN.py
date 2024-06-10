import torch
from torch import nn
import hiddenlayer as hl
import torchvision.transforms as transforms
import hiddenlayer as hl
from torchviz import make_dot

from networks.ClassifierBasedNoiseSignal.DenoisingModel import DualAttentionBlock, ResidualRefinementBlock

class ClassifierNE(nn.Module):
    def __init__(self, in_channels, input_dim=256*256*3, num_classes=2):
        super(ClassifierNE, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 2024),
            nn.BatchNorm1d(2024),
            nn.ReLU(),
            nn.Linear(2024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        return self.classifier(x)

if __name__ == '__main__':
    # Создаем экземпляр модели
    model = ResidualRefinementBlock(3, 3)  
    model.eval()
    # Создаем пример входных данных
    example_input = torch.randn(1, 3, 256, 256)  # Замените на ваши входные данные

   # Пропустите тестовый тензор через модель
    output_tensor = model(example_input)
    
    # Визуализируйте модель
    dot = make_dot(output_tensor, params=dict(model.named_parameters()))
    dot.node_attr.update(fontsize='50')
    dot.render("ResidualRefinementBlock", format="png")