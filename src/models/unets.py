import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=None):
        super(UNet, self).__init__()
        #self.num_channels = out_channels
        # Define a function that optionally adds Dropout to the conv block
        def conv_block(in_c, out_c, dropout=None):
            layers = [
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            ]
            # Add dropout if a level is specified
            if dropout:
                layers.append(nn.Dropout2d(p=dropout))
            return nn.Sequential(*layers)
        
        self.encoder1 = conv_block(in_channels, 64, dropout)
        self.encoder2 = conv_block(64, 128, dropout)
        self.encoder3 = conv_block(128, 256, dropout)
        self.encoder4 = conv_block(256, 512, dropout)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = conv_block(512, 1024, dropout)
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = conv_block(1024, 512, dropout)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256, dropout)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128, dropout)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64, dropout)
        
        self.output_layer = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        
        b = self.bottleneck(self.pool(e4))
        
        d4 = self.upconv4(b)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)
        
        return self.output_layer(d1)



class MultiUNet(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=None):
        super(MultiUNet, self).__init__()
        # Check if out_channels is a list or int
        if isinstance(out_channels, int):
            self.out_channels = [out_channels]
        elif isinstance(out_channels, list):
            self.out_channels = out_channels
        else:
            raise ValueError("out_channels must be an int or a list of ints.")

        def conv_block(in_c, out_c, dropout=None):
            layers = [
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            ]
            if dropout:
                layers.append(nn.Dropout2d(p=dropout))
            return nn.Sequential(*layers)
        
        self.encoder1 = conv_block(in_channels, 64, dropout)
        self.encoder2 = conv_block(64, 128, dropout)
        self.encoder3 = conv_block(128, 256, dropout)
        self.encoder4 = conv_block(256, 512, dropout)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = conv_block(512, 1024, dropout)
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = conv_block(1024, 512, dropout)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256, dropout)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128, dropout)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64, dropout)
        
        # Create an output layer for each element in out_channels
        self.output_layers = nn.ModuleList([
            nn.Conv2d(64, out_c, kernel_size=1) for out_c in self.out_channels
        ])
    
    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        
        b = self.bottleneck(self.pool(e4))
        
        d4 = self.upconv4(b)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)
        
        # Compute all outputs if multiple output layers are defined
        outputs = [layer(d1) for layer in self.output_layers]
        
        # Return a single tensor if only one output, otherwise return a list of tensors
        return outputs[0] if len(outputs) == 1 else outputs
    


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet34_Weights


class UNetResNet34(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetResNet34, self).__init__()
        
        # Load ResNet34 with updated 'weights' parameter
        self.resnet = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        
        # Modify the first convolutional layer to accept in_channels.
        #Note: this discards the pretrained weights in this layer.
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Encoder (ResNet's feature maps)
        self.encoder1 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool)
        self.encoder2 = self.resnet.layer1
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = self.conv_block(512, 256)
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(256, 128)
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(128, 64)
        
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=1, stride=1)
        self.decoder1 = self.conv_block(128, 64)
        
        # Output layer with final interpolation to ensure output matches input size
        self.output_layer = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder path
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        

        # Decoder path with upsampling and concatenation
        d4 = self.upconv4(e5)
        d4 = torch.cat([d4, e4], dim=1)  # Concatenate with corresponding encoder layer
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)  # Concatenate with corresponding encoder layer
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)
        
        # Final output with interpolation to match input size
        output = self.output_layer(d1)
        output = F.interpolate(output, x.shape[2:], mode='bilinear', align_corners=True)  # Ensure output size matches input size
        
        return output
    
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights

class UNetEfficientNetB0(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetEfficientNetB0, self).__init__()

        # Load pretrained EfficientNet-B0 weights
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.efficientnet = models.efficientnet_b0(weights=weights)

        # Replace the first conv layer to accommodate different in_channels
        self.efficientnet.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)

        # Get pretrained weights for other layers
        pretrained_dict = {k: v for k, v in self.efficientnet.state_dict().items() if not k.startswith('features.0.0')}
        self.efficientnet.load_state_dict(pretrained_dict, strict=False)

        # Define upsampling (transpose conv) and decoder blocks with aligned input/output channels
        # Define the UNet-like decoder part, adjust input/output channels based on EfficientNet feature maps
        self.upconv5 = nn.ConvTranspose2d(112, 112, kernel_size=1, stride=1)  # 14x14
        self.decoder5 = self.conv_block(192, 112)  # 112 is the output channel for e5

        self.upconv4 = nn.ConvTranspose2d(112, 40, kernel_size=2, stride=2)  # Upsample to 28x28
        self.decoder4 = self.conv_block(80, 80)  # 80 is the output channel for e4
        
        self.upconv3 = nn.ConvTranspose2d(80, 24, kernel_size=2, stride=2)  # Upsample to 56x56
        self.decoder3 = self.conv_block(48, 40)  # 40 is the output channel for e3

        self.upconv2 = nn.ConvTranspose2d(40, 16, kernel_size=2, stride=2)  # Upsample to 112x112
        self.decoder2 = self.conv_block(32, 24)  # 24 is the output channel for e2

        self.upconv1 = nn.ConvTranspose2d(24, 16, kernel_size=1, stride=1)  # Upsample to 112x112
        self.decoder1 = self.conv_block(48, 16)  # 16 is the output channel for e1

        self.final_upconv = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)  # Upsample from 112x112 to 224x224
        self.final_decoder = self.conv_block(16, 16)  # Optional additional processing after upsampling

        # Final output layer
        self.output_layer = nn.Conv2d(16, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder (EfficientNet-B0's feature maps)
        e1 = self.efficientnet.features[0](x)  # Output: [batch_size, 32, H/2, W/2]
        e2 = self.efficientnet.features[1](e1)  # Output: [batch_size, 16, H/4, W/4]
        e3 = self.efficientnet.features[2](e2)  # Output: [batch_size, 24, H/8, W/8]
        e4 = self.efficientnet.features[3](e3)  # Output: [batch_size, 40, H/16, W/16]
        e5 = self.efficientnet.features[4](e4)  # Output: [batch_size, 80, H/32, W/32]
        e6 = self.efficientnet.features[5](e5)  # Output: [batch_size, 112, H/32, W/32]
        
        # Decoder (Upsampling)
        d5 = self.upconv5(e6)  # Output: [batch_size, 80, H/16, W/16]
        d5 = torch.cat([d5, e5], dim=1)  # Concatenate with corresponding encoder layer
        d5 = self.decoder5(d5)
        
        d4 = self.upconv4(d5)  # Output: [batch_size, 40, H/8, W/8]
        d4 = torch.cat([d4, e4], dim=1)  # Concatenate with corresponding encoder layer
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)  # Output: [batch_size, 24, H/4, W/4]
        d3 = torch.cat([d3, e3], dim=1)  # Concatenate with corresponding encoder layer
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)  # Output: [batch_size, 16, H/2, W/2]
        d2 = torch.cat([d2, e2], dim=1)  # Concatenate with corresponding encoder layer
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)  # Output: [batch_size, 32, H, W]
        d1 = torch.cat([d1, e1], dim=1)  # Concatenate with corresponding encoder layer
        d1 = self.decoder1(d1)

        # Final upsampling step to match the input size (224x224)
        d_final = self.final_upconv(d1)
        d_final = self.final_decoder(d_final)

        # Output layer
        return self.output_layer(d_final)



import torchvision.models as models

class UNetConvNext(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetConvNext, self).__init__()
        self.convnext = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        # Modify the first convolutional layer
        self.convnext.features[0][0] = nn.Conv2d(in_channels, 96, kernel_size=4, stride=4)

        # Encoder
        self.encoder1 = self.convnext.features[0]
        self.encoder2 = self.convnext.features[1]
        self.encoder3 = self.convnext.features[2]
        self.encoder4 = self.convnext.features[3]
        self.encoder5 = self.convnext.features[4]

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(384, 384, kernel_size=2, stride=2)
        self.decoder4 = self.conv_block(576, 384) #576 = 384+192

        self.upconv3 = nn.ConvTranspose2d(384, 192, kernel_size=1, stride=1)
        self.decoder3 = self.conv_block(384, 192)

        self.upconv2 = nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(192, 96)

        self.upconv1 = nn.ConvTranspose2d(96, 96, kernel_size=1, stride=1)
        self.decoder1 = self.conv_block(192, 96)

        self.final_upconv = nn.ConvTranspose2d(96, 96, kernel_size=4, stride=4)  # Upsample from 112x112 to 224x224
        self.final_decoder = self.conv_block(96, 96)  # Optional additional processing after upsampling


        self.output_layer = nn.Conv2d(96, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def _load_pretrained_weights(self):
        print("Loading pretrained weights...")
        # Get the pre-trained state dict
        original_state_dict = self.convnext.state_dict()

        # Get the current model's state dict
        current_state_dict = self.state_dict()

        # Filter the original state dict to match keys in the current model's state dict
        filtered_state_dict = {k: v for k, v in original_state_dict.items() if k in current_state_dict}

        # Load the filtered state dict with strict=False to ignore unmatched keys
        self.load_state_dict(filtered_state_dict, strict=False)
        print("Filtered state_dict keys:", filtered_state_dict.keys())


    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        d4 = self.upconv4(e5)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)

        d_final = self.final_upconv(d1)
        d_final = self.final_decoder(d_final)

        return self.output_layer(d_final)



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

    model = UNet(12, out_channels=5).to(device)  # 12 channels, 5 classes
    input_tensor = torch.rand(6, 12, 64, 64).to(device)  # batch = 6
    output = model(input_tensor)
    print('-------------------------------------------')
    print('UNet shapes:')
    print('Input:', input_tensor.shape)
    print('Output:', output.shape)
    print_gpu_memory()
    torch.cuda.empty_cache() 

    model = UNetResNet34(12, out_channels=5).to(device)  # 12 channels, 5 classes
    input_tensor = torch.rand(6, 12, 224, 224).to(device)  # batch = 6
    output = model(input_tensor)
    print('-------------------------------------------')
    print('UNetResNet34 shapes:')
    print('Input:', input_tensor.shape)
    print('Output:', output.shape)
    print_gpu_memory()
    torch.cuda.empty_cache() 
    
    model = UNetEfficientNetB0(12, out_channels=5).to(device)
    input_tensor = torch.rand(6, 12, 224, 224).to(device)  # batch = 6
    output = model(input_tensor)
    print('-------------------------------------------')
    print('UNetEfficientNetB0 shapes:')
    print('Input:', input_tensor.shape)
    print('Output:', output.shape)
    print_gpu_memory()
    torch.cuda.empty_cache() 

    model = UNetConvNext(12, out_channels=5).to(device)
    input_tensor = torch.rand(6, 12, 224, 224).to(device)  # batch = 6
    output = model(input_tensor)
    print('-------------------------------------------')
    print('UNetConvNext shapes:')
    print('Input:', input_tensor.shape)
    print('Output:', output.shape)
    print_gpu_memory()
    torch.cuda.empty_cache() 

