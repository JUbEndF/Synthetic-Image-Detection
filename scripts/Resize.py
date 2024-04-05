import os
from PIL import Image

def resize_images_in_folder(input_folder, output_folder, new_width, new_height):
    """
    Уменьшает разрешение всех изображений в заданной папке без потери качества и сохраняет их в другую папку.
    
    Args:
        input_folder (str): Путь к папке с исходными изображениями.
        output_folder (str): Путь к папке, в которой будут сохранены уменьшенные изображения.
        new_width (int): Новая ширина изображения.
        new_height (int): Новая высота изображения.
    """
    # Проверяем, существует ли папка для сохранения изображений, и создаем ее, если необходимо
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Перебираем все файлы в папке с исходными изображениями
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):  # Проверяем расширение файла
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            resize_image(input_path, output_path, new_width, new_height)

def resize_image(input_path, output_path, new_width, new_height):
    """
    Уменьшает разрешение изображения без потери качества.
    
    Args:
        input_path (str): Путь к исходному изображению.
        output_path (str): Путь к сохраненному уменьшенному изображению.
        new_width (int): Новая ширина изображения.
        new_height (int): Новая высота изображения.
    """
    # Открываем изображение
    image = Image.open(input_path)
    
    # Масштабируем изображение
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Сохраняем уменьшенное изображение
    resized_image.save(output_path)

# Пример использования
input_folder = "D:/Datasets/07000-20240403T135713Z-001/07000"
output_folder = "D:/Datasets/07000-20240403T135713Z-001/07000_512"
new_width = 512
new_height = 512
resize_images_in_folder(input_folder, output_folder, new_width, new_height)