import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import torch
import torch.nn as nn
import timm

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=None):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

class HRNetSegmentation(nn.Module):
    def __init__(self, in_channels, num_classes, backbone="hrnet_w18", pretrained=True, debug = False):
        """
        Initialize HRNet for segmentation.

        Parameters:
        num_classes (int): Number of output classes.
        backbone (str): HRNet backbone type (e.g., "hrnet_w18", "hrnet_w32", "hrnet_w48").
        pretrained (bool): Whether to use a pre-trained backbone.
        in_channels (int): Number of input channels (default: 12).
        """
        super(HRNetSegmentation, self).__init__()
        self.debug = debug
        # Load HRNet backbone from timm
        self.backbone = timm.create_model(backbone, features_only=True, pretrained=pretrained)

        # Modify the first convolutional layer to accept any number of input channels
        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                in_channels,
                self.backbone.conv1.out_channels,
                kernel_size=self.backbone.conv1.kernel_size,
                stride=self.backbone.conv1.stride,
                padding=self.backbone.conv1.padding,
                bias=self.backbone.conv1.bias is not None,
            )

        
        feature_channels = self.backbone.feature_info.channels()[0]
        self.last_up = nn.ConvTranspose2d(feature_channels, feature_channels, kernel_size=2, stride=2)
        self.output_layer = nn.Conv2d(feature_channels, num_classes, kernel_size=1)
        
    def forward(self, x):
        """
        Forward pass.

        Parameters:
        x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
        torch.Tensor: Output tensor of shape (N, num_classes, H, W).
        """
        # Extract features from the backbone
        features = self.backbone(x)  # List of feature maps from different resolutions
        output = self.output_layer(self.last_up(features[0]))
        return output

class HRNetSegmentation_(nn.Module):
    def __init__(self, in_channels, num_classes, backbone='hrnet_w18', pretrained=True, debug=False):
        super(HRNetSegmentation, self).__init__()
        
        # Load the HRNet backbone
        self.backbone = timm.create_model(backbone, features_only=True, pretrained=pretrained)
        
        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                in_channels,
                self.backbone.conv1.out_channels,
                kernel_size=self.backbone.conv1.kernel_size,
                stride=self.backbone.conv1.stride,
                padding=self.backbone.conv1.padding,
                bias=self.backbone.conv1.bias is not None,
            )
        # Get the number of output channels from the backbone
        self.channels = self.backbone.feature_info.channels()
        
        # Transposed convolution layers for upsampling and feature fusion
        self.up1 = nn.ConvTranspose2d(self.channels[-1], self.channels[-2], kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(self.channels[-2], self.channels[-3], kernel_size=4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(self.channels[-3], self.channels[-4], kernel_size=4, stride=2, padding=1)
        self.up4 = nn.ConvTranspose2d(self.channels[-4], self.channels[-5], kernel_size=4, stride=2, padding=1)
        self.up5 = nn.ConvTranspose2d(self.channels[-5], self.channels[-5], kernel_size=4, stride=2, padding=1)
        
        self.conv1 = ConvBlock(self.channels[-2], self.channels[-2])
        self.conv2 = ConvBlock(self.channels[-3], self.channels[-3])
        self.conv3 = ConvBlock(self.channels[-4], self.channels[-4])
        self.conv4 = ConvBlock(self.channels[-5], self.channels[-5])
        
        # Final layer to produce segmentation map
        self.final_conv = ConvBlock(self.channels[-5], num_classes)
        
    def forward(self, x):
        # Extract multi-scale features from HRNet
        features = self.backbone(x)
        x = self.up1(features[-1]) + features[-2]
        x = self.conv1(x)
        x = self.up2(x) + features[-3]
        x = self.conv2(x)
        x = self.up3(x) + features[-4]
        x = self.conv3(x)
        x = self.up4(x) + features[-5]
        x = self.conv4(x)
        x = self.up5(x)
        # Final convolution to produce segmentation map
        x = self.final_conv(x)
        return x

# Testing
if __name__ == "__main__":

    def print_gpu_memory(prefix=""):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            print(f"{prefix} Memory Allocated: {allocated:.2f} MB")
            print(f"{prefix} Memory Reserved: {reserved:.2f} MB")
        else:
            print("CUDA is not available.")
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache() 

    print(device)
    
    # Example usage
    model = HRNetSegmentation(in_channels = 12, num_classes=9, backbone="hrnet_w18", pretrained=True)
    input_tensor = torch.randn(16, 12, 512, 512)  # Batch of 2, 3 channels, 512x512 images
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")  # Expected shape: (2, 21, 512, 512)
    if 0:

        # Example usage
        model = HRNetSegmentation(in_channels = 12, num_classes=9, backbone="hrnet_w32", pretrained=True)
        input_tensor = torch.randn(16, 12, 512, 512)  # Batch of 2, 3 channels, 512x512 images
        output = model(input_tensor)
        print(f"Output shape: {output.shape}")  # Expected shape: (2, 21, 512, 512)

        # Example usage
        model = HRNetSegmentation(in_channels = 12, num_classes=9, backbone="hrnet_w48", pretrained=True)
        input_tensor = torch.randn(16, 12, 512, 512)  # Batch of 2, 3 channels, 512x512 images
        output = model(input_tensor)
        print(f"Output shape: {output.shape}")  # Expected shape: (2, 21, 512, 512)
