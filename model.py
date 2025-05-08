import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ResNet18Segmentation(nn.Module):
    def __init__(self, num_classes=7, output_size=(150, 150)):
        super(ResNet18Segmentation, self).__init__()
        
        # Store output size
        self.output_size = output_size
        
        # Load pretrained ResNet18
        resnet18 = models.resnet18(pretrained=True)
        
        # Encoder: Use all ResNet18 layers except the final fc layer
        self.encoder_conv1 = resnet18.conv1
        self.encoder_bn1 = resnet18.bn1
        self.encoder_relu = resnet18.relu
        self.encoder_maxpool = resnet18.maxpool
        self.encoder_layer1 = resnet18.layer1
        self.encoder_layer2 = resnet18.layer2
        self.encoder_layer3 = resnet18.layer3
        self.encoder_layer4 = resnet18.layer4
        
        # Get the number of features from the last ResNet layer
        self.in_channels = resnet18.layer4[-1].conv2.out_channels  # 512 for ResNet18
        
        # Decoder
        self.decoder = nn.Sequential(
            # Upsample block 1
            nn.ConvTranspose2d(self.in_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Upsample block 2
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Upsample block 3
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Upsample block 4
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Final convolution to get desired number of classes
            nn.Conv2d(32, num_classes, kernel_size=3, padding=1)
        )
        
        # Initialize decoder weights
        self._initialize_weights()

    def forward(self, x):
        # Encoder
        x = self.encoder_conv1(x)
        x = self.encoder_bn1(x)
        x = self.encoder_relu(x)
        x = self.encoder_maxpool(x)
        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x)
        x = self.encoder_layer3(x)
        x = self.encoder_layer4(x)
        
        # Decoder
        x = self.decoder(x)
        
        # Final interpolation to match exact output size
        x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
        
        return x
    
    def _initialize_weights(self):
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
