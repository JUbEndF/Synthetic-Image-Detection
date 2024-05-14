import torch
from networks.modelCNN.AIDetectionCNNModel import AIDetectionCNN, AIDetectionCNN_split_Linear_layers, AIDetectionCNN_split_Linear_layers_NormBatch, AIDetectionCNNBaseNormBatch
from torchsummary import summary
import torchvision.transforms as transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model = AIDetectionCNN(input_channels=3, output_size=2)
summary(model, input_size=(3, 256, 256), device="cpu")
