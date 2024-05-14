import glob
import re

import torch

def loadModel(model, device):
    # Получаем список всех файлов моделей
    model_files = glob.glob(f"./saveModel/AIDetectionModels/{type(model).__name__}/{type(model).__name__}_epoch_*.pth")
    
    # Если есть сохраненные модели
    if model_files:
        # Извлекаем номера эпох из имен файлов
        epoch_numbers = [int(re.search(r'\d+', file.split('_')[-1]).group()) for file in model_files]
        
        # Выбираем номер стартовой эпохи как максимальный номер плюс один
        start_epoch = max(epoch_numbers)
        
        # Формируем путь к файлу последней модели
        last_model_path = f"./saveModel/AIDetectionModels/{type(model).__name__}/{type(model).__name__}_epoch_{start_epoch}.pth"
        
        # Загружаем последнюю сохраненную модель
        model = torch.load(last_model_path)
        model.to(device)
        
        print(f"Loaded model from {last_model_path}")
        return model, start_epoch
    else:
        print("No saved models found.")
        return model, 0