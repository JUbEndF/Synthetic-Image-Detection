from keras.preprocessing.image import ImageDataGenerator
import torch
from torchvision import datasets


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


