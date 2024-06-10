from PIL import Image, ImageFilter
import torch
import numpy as np
import cv2
from torchvision import transforms
import scipy.fftpack as fftpack
from networks.ClassifierBasedNoiseSignal.ClassifierNoiseExtraction import DenoisingNetwork

def compute_amplitude_and_average(complex_array: np.ndarray, axis: int):
    """
    Функция для вычисления амплитуды комплексных чисел в комплексном массиве
    и последующего усреднения по указанной оси.

    Аргументы:
    complex_array (np.ndarray): Комплексный массив, в котором каждый элемент представляет собой комплексное число.
    axis (int): Ось, по которой будет выполнено усреднение. 0 - усреднение по столбцам, 1 - усреднение по строкам.

    Возвращает:
    np.ndarray: Результат усреднения амплитуд по указанной оси.
    """

    # Выполняем FFT для получения комплексного массива
    fft_array = fftpack.fft2(complex_array)

    # Вычисляем амплитуду для каждого комплексного числа
    amplitude = np.abs(fft_array)

    # Выполняем усреднение по указанной оси
    averaged_amplitude = np.mean(amplitude, axis=axis)

    return averaged_amplitude

# Функция для обработки изображения с помощью модели
def process_image(image, model, device):

    # Преобразование изображения в тензор и нормализация
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    tensor_image = transform(image).unsqueeze(0).to(device)  # Добавляем размерность пакета и отправляем на устройство
    model.to(device)
    # Применение модели к тензору изображения
    with torch.no_grad():
        model.eval()
        output_tensor = model(tensor_image)
    
    # Преобразование тензора обратно в изображение
    output_image = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output_image = np.clip(output_image * 0.5 + 0.5, 0.0, 1.0) * 255  # Раскодирование и приведение к [0, 255]
    output_image = output_image.astype(np.uint8)

    return output_image

def dftImage(image_path, device):
    image = Image.open(image_path).convert('RGB')

    # Загрузка модели
    model = DenoisingNetwork().to('cuda' if torch.cuda.is_available() else 'cpu')

    # Обработка изображения с помощью модели
    denoise = process_image(image, model, device)

    noise = image - denoise

    b, g, r = cv2.split(noise)

    # Применение преобразования Фурье к каждому каналу
    dft_b = compute_amplitude_and_average(b, 0)
    dft_g = compute_amplitude_and_average(g, 0)
    dft_r = compute_amplitude_and_average(r, 0)



    return [dft_b, dft_g, dft_r]






