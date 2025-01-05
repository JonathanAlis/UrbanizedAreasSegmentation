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


def fuse_features(features, reduced_channels=64, debug = False):
    """
    Fuse feature maps incrementally after reducing channels.

    Parameters:
    features (list): List of feature maps from different resolutions.
    reduced_channels (int): Number of channels to reduce each feature map to.

    Returns:
    torch.Tensor: Fused feature map.
    """
    # Reduce channels in each feature map
    reducers = nn.ModuleList([
        nn.Conv2d(f.shape[1], reduced_channels, kernel_size=1)
        for f in features
    ]).to(features[0].device)

    # Reduce channels and upsample to the highest resolution
    fused = reducers[0](features[0])
    for i, f in enumerate(features[1:]):
        if debug:
            print(f"Before reduction: {f.shape}")
        reduced_f = reducers[i + 1](f)  # Reduce channels
        if debug:
            print(f"After reduction: {reduced_f.shape}")
            print(f"Fused shape: {fused.shape}")
        
        # Upsample and add to the fused feature map
        fused = fused + F.interpolate(reduced_f, size=fused.shape[2:], mode="bilinear", align_corners=False)
    
    return fused



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

        # Segmentation head
        in_channels = self.backbone.feature_info.channels()[0]  # Use the first feature map's channels
        self.segmentation_head = nn.Conv2d(in_channels, num_classes, kernel_size=1)

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

        # Fuse feature maps incrementally
        fused_features = fuse_features(features)

        # Apply the segmentation head
        output = self.segmentation_head(fused_features)

        # Resize output to match input size (optional, depending on your use case)
        output = F.interpolate(output, size=x.size()[2:], mode="bilinear", align_corners=False)

        return output


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
