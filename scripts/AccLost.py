import re
import matplotlib.pyplot as plt
import argparse

def parse_file(file_path):
    # Инициализация списков для хранения значений
    epochs = []
    loss_values = []
    accuracy_values = []

    # Логическая переменная для игнорирования тестовых данных
    in_test_section = False

    # Открытие файла и чтение данных
    with open(file_path, 'r') as file:
        for line in file:
            # Определение начала секции тестовых данных
            if 'Test data acc' in line:
                in_test_section = True

            # Игнорирование строк в секции тестовых данных
            if in_test_section:
                # Проверка на новую эпоху после начала тестовой секции
                epoch_match = re.search(r'Epoch \[(\d+)/\d+\]', line)
                if epoch_match:
                    in_test_section = False
                if in_test_section:
                    continue
            
            # Поиск номера эпохи
            epoch_match = re.search(r'Epoch \[(\d+)/\d+\]', line)
            if epoch_match:
                epochs.append(int(epoch_match.group(1)))
            
           # Точное совпадение строки для Loss
            loss_match = re.match(r'Loss: ([\deE\+\-\.]+)', line.strip())
            if loss_match:
                loss_values.append(float(loss_match.group(1)))
            
            # Точное совпадение строки для accuracy
            accuracy_match = re.match(r'accuracy: ([\deE\+\-\.]+)', line.strip())
            if accuracy_match:
                accuracy_values.append(float(accuracy_match.group(1)))
    
    return epochs, loss_values, accuracy_values

def plot_metrics(epochs, loss_values, accuracy_values):
    # Построение двух графиков один под другим
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5))

    # График потерь
    ax1.plot(epochs, loss_values, 'o-', color='tab:red', label='Потери')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Потери')
    ax1.set_title('График потерь')
    ax1.grid(True)
    ax1.tick_params(labelsize=12)  # Установка размера шрифта меток на осях
    
    # График точности
    ax2.plot(epochs, accuracy_values, 'o-', color='tab:blue', label='Точность')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Точность')
    ax2.set_title('График точности')
    ax2.grid(True)
    ax2.tick_params(labelsize=12)  # Установка размера шрифта меток на осях

    fig.tight_layout()  # Для корректного отображения графиков
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Построение графиков потерь и точности обучения из файла логов.')
    parser.add_argument('file_path', type=str, help='Путь к файлу логов')

    args = parser.parse_args()

    epochs, loss_values, accuracy_values = parse_file(args.file_path)
    plot_metrics(epochs, loss_values, accuracy_values)

if __name__ == "__main__":
    main()