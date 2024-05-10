import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class DualAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DualAttentionBlock, self).__init__()
        # Слой свертки для обработки входа
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Механизм внимания
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.att_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.att_relu = nn.ReLU()
        self.att_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Свертка и активация
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        
        # Глобальное усреднительное пулирование
        avg_out = self.global_avg_pool(out)
        avg_out = self.att_conv1(avg_out)
        avg_out = self.att_relu(avg_out)
        avg_out = self.att_conv2(avg_out)
        
        # Глобальное максимальное пулирование
        max_out = self.global_max_pool(out)
        max_out = self.att_conv1(max_out)
        max_out = self.att_relu(max_out)
        max_out = self.att_conv2(max_out)

        # Механизм внимания
        att = self.sigmoid(avg_out * max_out)

        # Умножение весов внимания на исходные карты признаков
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
        self.conv_input = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.rrg_blocks = nn.ModuleList([ResidualRefinementBlock(32, 32) for _ in range(num_rrg_blocks)])
        self.conv_output = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv_input(x)
        for rrg_block in self.rrg_blocks:
            out = rrg_block(out)
        out = self.conv_output(out)
        return out
    
class NoiseExtractionNetwork(nn.Module):
    def __init__(self, in_channels=3, num_rrg_blocks=4):
        super(NoiseExtractionNetwork, self).__init__()
        self.conv_input = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.rrg_blocks = nn.ModuleList([ResidualRefinementBlock(32, 32) for _ in range(num_rrg_blocks)])
        self.conv_output = nn.Conv2d(32, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Сохраняем оригинальное входное изображение
        original_input = x.clone()
        
        # Обработка входного изображения
        out = self.conv_input(x)
        for rrg_block in self.rrg_blocks:
            out = rrg_block(out)
        
        # Получаем деноизированное изображение
        denoised_output = self.conv_output(out)
        
        # Вычисляем шум путем вычитания деноизированного изображения из оригинального входного изображения
        noise_output = original_input - denoised_output
        
        return noise_output