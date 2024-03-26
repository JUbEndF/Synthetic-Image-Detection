import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class DualAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DualAttentionBlock, self).__init__()
        # Convolution layers to process the input
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # Global attention mechanism
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.att_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.att_relu = nn.ReLU()
        self.att_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Convolutional operations
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        
        # Global average pooling
        avg_out = self.global_avg_pool(out)
        avg_out = self.att_conv1(avg_out)
        avg_out = self.att_relu(avg_out)
        avg_out = self.att_conv2(avg_out)
        
        # Global max pooling
        max_out = self.global_max_pool(out)
        max_out = self.att_conv1(max_out)
        max_out = self.att_relu(max_out)
        max_out = self.att_conv2(max_out)
        
        # Attention mechanism
        att = self.sigmoid(avg_out * max_out)
        
        # Multiply attention weights with the original feature maps
        out = out * att
        
        return out
    
class ResidualRefinementBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualRefinementBlock, self).__init__()
        self.dab_block1 = DualAttentionBlock(in_channels, out_channels)
        self.dab_block2 = DualAttentionBlock(out_channels, out_channels)
        self.conv_merge = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x):
        dab_output1 = self.dab_block1(x)
        dab_output2 = self.dab_block2(dab_output1)

        # Производим конкатенацию выходов DAB блоков
        merged_output = torch.cat([dab_output1, dab_output2], dim=1)

        # Производим свертку для объединения выходов
        out = self.conv_merge(merged_output)
        return out
    
class DenoisingNetwork(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_rrg_blocks=4):
        super(DenoisingNetwork, self).__init__()
        self.conv_input = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.rrg_blocks = nn.ModuleList([ResidualRefinementBlock(64, 64) for _ in range(num_rrg_blocks)])
        self.conv_output = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv_input(x)
        for rrg_block in self.rrg_blocks:
            out = rrg_block(out)
        out = self.conv_output(out)
        return out
    
    def save_model(self, epoch, dir_path):
        # Создание директории, если её нет
        os.makedirs(dir_path, exist_ok=True)
        # Генерация имени файла
        file_name = os.path.join(dir_path, f'Denoising_RRG_model_epoch_{epoch}.pth')
        torch.save(self.state_dict(), file_name)

    @classmethod
    def load_model(cls, file_path):
        model = cls()
        model.load_state_dict(torch.load(file_path))
        model.eval()  # Установите в режим оценки (evaluation), если это необходимо
        return model