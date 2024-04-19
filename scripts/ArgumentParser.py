import argparse

class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Neural Network Training Parameters')
        self.add_arguments()

    def add_arguments(self):
        # Добавление аргументов
        self.parser.add_argument('--device', type=str, default='cuda', help='Device for training: cpu or cuda')
        self.parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
        self.parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
        self.parser.add_argument('--train_data_path', type=str, default='./data/MyDataset/train', help='Path to training data')
        self.parser.add_argument('--val_data_path', type=str, default='./data/MyDataset/test', help='Path to validation data')
        self.parser.add_argument('--test_data_path', type=str, default='./data/MyDataset/test', help='Path to test data')
        self.parser.add_argument('--model_save_path', type=str, default='./saved_models', help='Directory to save trained models')
        self.parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for SGD optimizer')
        self.parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay coefficient')
        # Добавьте здесь другие параметры, если необходимо
        
    def parse_args(self):
        # Парсинг аргументов командной строки
        return self.parser.parse_args()