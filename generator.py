import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    'skip-connection layers for high-fidelity image synthesis'
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        # THE SKIP CONNECTION: Adding original input 'x' to the output
        return x + self.conv_block(x)

class Generator(nn.Module):
    """
    'feature extraction pipeline combining deep CNNs'
    """
    def __init__(self, scale_factor=2): # 2x scale roughly aligns with ~125% enhancement goals
        super(Generator, self).__init__()
        
        # Initial Feature Extraction
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )
        
        # Deep CNN Pipeline with 5 Skip-Connection Blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(5)]
        )
        
        # Mid-level Feature CNN
        self.mid_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        
        # Upsampling block to achieve the required resolution increase
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.PReLU()
        )
        
        # Final Output Layer
        self.final_conv = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        initial_features = self.initial_conv(x)
        res_features = self.residual_blocks(initial_features)
        
        # Another skip connection across the entire residual pipeline
        mid_features = initial_features + self.mid_conv(res_features) 
        
        upsampled_features = self.upsample(mid_features)
        return (torch.tanh(self.final_conv(upsampled_features)) + 1) / 2
