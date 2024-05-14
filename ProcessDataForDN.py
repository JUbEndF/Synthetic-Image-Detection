import os
from skimage import io, img_as_float
from torch.utils.data import DataLoader
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from networks.ClassifierBasedNoiseSignal.ClassifierNoiseExtraction import DenoisingNetwork
from scripts.ArgumentParser import ArgumentParser


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    args = arg_parser.parse_args()
    transform = transforms.ToTensor()

    # Подготовьте загрузчик данных
    

    # Переведите модель на нужное устройство (например, CUDA, если доступно)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    denoising_network = DenoisingNetwork().to('cuda' if torch.cuda.is_available() else 'cpu')

    data_loaders = list()
    dir_data = "./data/progan_train_DN"
    # Получаем список всех элементов в директории
    all_items = os.listdir(dir_data)
    print(all_items)
    # Создаем загрузчики данных для каждой категории и класса
    i = 0
    for class_name in [item for item in all_items if os.path.isdir(os.path.join(dir_data, item))]:  # добавьте все классы
        print(class_name)
        data_folder = os.path.join(dir_data, class_name)
        dataset = ImageFolder(root=data_folder, transform=transform)
    
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)


        # Обработайте изображения в batch
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(dataloader):
                # Переместите изображения на устройство
                images = images.to(device)
                
                # Примените вашу модель шумоподавления
                denoised_images = denoising_network(images)
                
                # Вычислите шум
                noise = images - denoised_images
                
                # Замените исходные файлы изображений шумом
                for i, img_noise in enumerate(noise):
                    # Получите имя файла исходного изображения
                    image_filename = dataloader.dataset.samples[i + batch_idx * len(images)][0]
                    
                    # Преобразуйте тензор шума в формат, совместимый с skimage
                    # Например, используйте np.squeeze для удаления лишних размерностей,
                    # затем преобразуйте тензор в numpy-массив и перенесите оси.
                    img_noise_np = np.squeeze(img_noise.cpu().numpy()).transpose(1, 2, 0)
                    
                    # Убедитесь, что тип данных изображения соответствует требуемому формату
                    # Преобразуйте значения к диапазону [0,1] и типу float32 или к диапазону [0,255] и типу uint8.
                    if img_noise_np.dtype == np.float32:
                        img_noise_np = np.clip(img_noise_np, 0, 1)
                    elif img_noise_np.dtype == np.uint8:
                        img_noise_np = np.clip(img_noise_np * 255, 0, 255).astype(np.uint8)

                    # Сохраните шумовое изображение, заменив исходный файл
                    io.imsave(image_filename, img_noise_np)

