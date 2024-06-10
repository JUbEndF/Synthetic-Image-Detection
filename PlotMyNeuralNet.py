import torch
from networks.modelCNN.AIDetectionCNNModel import AIDetectionCNN
import hiddenlayer as hl
import torchvision.transforms as transforms
import torch
import torch
import hiddenlayer as hl
from torchviz import make_dot

if __name__ == '__main__':
    # Создаем экземпляр модели
    model = AIDetectionCNN(input_channels=3, output_size=2)  
    model.eval()
    # Создаем пример входных данных
    example_input = torch.randn(1, 3, 256, 256)  # Замените на ваши входные данные

   # Пропустите тестовый тензор через модель
    output_tensor = model(example_input)
    
    # Визуализируйте модель
    dot = make_dot(output_tensor, params=dict(model.named_parameters()))
    dot.node_attr.update(fontsize='50')
    dot.render("AIDetectionCNN", format="png")