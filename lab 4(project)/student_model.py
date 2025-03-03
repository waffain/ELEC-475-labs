import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

class LightweightDeepLabV3(nn.Module):
    def __init__(self, num_classes=21, pretrained=True):
        super().__init__()
        
        # Load the pretrained model
        base_model = deeplabv3_mobilenet_v3_large(pretrained=pretrained)
        
        # Get the backbone
        self.backbone = base_model.backbone
        
        # Create a lighter classifier
        self.classifier = nn.Sequential(
            # Reduce input channels from backbone
            nn.Conv2d(960, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Simplified ASPP
            nn.Conv2d(256, 256, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Dropout for regularization
            nn.Dropout(0.1),
            
            # Final classification layer
            nn.Conv2d(256, num_classes, 1)
        )
        
        # Initialize weights for the new layers
        self._init_weight()

    def forward(self, x):
        input_shape = x.shape[-2:]
        
        # Get features from backbone
        features = self.backbone(x)['out']
        
        # Apply classifier
        x = self.classifier(features)
        
        # Upsample to input resolution
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        
        return x
    
    def _init_weight(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)