import os
import argparse
import random
import shutil

def split_dataset(input_folder, output_folder, split_ratio, folder_name):
    # Создаем папки для обучающей и тестовой выборок, если они еще не существуют
    train_folder = os.path.join(output_folder, "train")
    test_folder = os.path.join(output_folder, "test")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    
    train_folder = os.path.join(train_folder, folder_name)
    test_folder = os.path.join(test_folder, folder_name)
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Получаем список файлов изображений в папке
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg') or f.endswith('.png')]
    # Перемешиваем список
    random.shuffle(image_files)
    
    # Вычисляем количество изображений для обучающей и тестовой выборок
    num_train = int(len(image_files) * split_ratio)
    num_test = len(image_files) - num_train
    
    # Копируем изображения в соответствующие папки
    for i, image_file in enumerate(image_files):
        src_path = os.path.join(input_folder, image_file)
        if i < num_train:
            dst_path = os.path.join(train_folder, image_file)
        else:
            dst_path = os.path.join(test_folder, image_file)
        shutil.copyfile(src_path, dst_path)
        
    print("Разделение завершено. Обучающая выборка содержит", num_train, "изображений, а тестовая выборка содержит", num_test, "изображений.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split images into train and test sets.")
    parser.add_argument("--input_folder", default="D:\\Datasets\\Kandinsry\\Landscapes", help="Path to the folder containing images.")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Ratio of training set size to test set size.")
    parser.add_argument("--folder_name", default="fake", help="Сlass name images.")
    args = parser.parse_args()

    output_folder = os.path.dirname(args.input_folder)

    split_dataset(args.input_folder, output_folder, args.split_ratio, args.folder_name)