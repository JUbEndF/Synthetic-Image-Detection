from tqdm import tqdm
from scripts.ArgumentParser import ArgumentParser
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from transformers import CLIPProcessor, CLIPModel, ViTImageProcessor, ViTModel
import torchvision.transforms as transforms


model_dict = {
    "openai/clip-vit-base-patch32": (CLIPModel, CLIPProcessor),
    "openai/clip-resnet-50": (CLIPModel, CLIPProcessor),
    "google/vit-base-patch16-224": (ViTModel, ViTImageProcessor),
    "google/vit-large-patch16-224": (ViTModel, ViTImageProcessor)
}

def load_data(pathDir: str):
    # Список для хранения загрузчиков данных
    data_loaders = list()

    # Получаем список всех элементов в директории
    all_items = os.listdir(pathDir)

    # Создаем загрузчики данных для каждой категории и класса
    for class_name in [item for item in all_items if os.path.isdir(os.path.join(pathDir, item))]:  # добавьте все классы
        data_folder = os.path.join(pathDir, class_name)
        dataset = ImageFolder(root=data_folder, transform=transform)
        data_loaders.append(dataset)

    dataset = torch.utils.data.ConcatDataset(data_loaders)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

if __name__ == '__main__':
    arg_parser = ArgumentParser()
    args = arg_parser.parse_args()


    model_name = "google/vit-base-patch16-224"
    model_class, processor_class = model_dict[model_name]

    transform = transforms.ToTensor()

    # Подготовьте загрузчик данных
    dataloader = load_data(args.val_data_path)

    # Загрузите модель CLIP и процессор
    model = model_class.from_pretrained(model_name)
    processor = processor_class.from_pretrained(model_name)

    # Переведите модель на нужное устройство (например, CUDA, если доступно)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Переведите модель в режим оценки (evaluation)
    model.eval()

    # Списки для хранения векторов признаков и меток классов
    features = []
    labels = []

    # Извлечение векторов признаков
    with torch.no_grad():
        for images, target_labels in tqdm(dataloader):
            # Переместите данные на нужное устройство
            
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)

            # Получение признаков с помощью модели
            if isinstance(model, CLIPModel):
                images = model.get_image_features(pixel_values)
            elif isinstance(model, ViTModel):
                outputs = model(pixel_values)
                images = outputs.last_hidden_state.mean(dim=1)

            
            # Переведите векторы признаков в список и переместите на CPU
            features.append(images.cpu())
            
            # Сохраняйте метки классов
            labels.append(target_labels)

    # Конвертируем списки в тензоры
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    # Сохранение векторов признаков и меток классов в файл .npz
    output_path = f"clip_features_val.npz"
    # Загрузка существующих данных из файла .npz
    if os.path.exists(output_path):
        existing_data = np.load(output_path, mmap_mode="r+")

        # Получите существующие векторы признаков и метки классов
        existing_features = existing_data["features"]
        existing_labels = existing_data["labels"]

        # Соедините старые и новые данные
        features = np.concatenate([existing_features, features.numpy()], axis=0)
        labels = np.concatenate([existing_labels, labels.numpy()], axis=0)
        print(f"Файл {output_path} успешно загружен.")
        print(f"Размер векторов признаков: {features.shape}")
        print(f"Размер меток классов: {labels.shape}")

    np.savez(output_path, features=features.numpy(), labels=labels.numpy())

    print(f"Векторы признаков и метки классов сохранены в файл: {output_path}")