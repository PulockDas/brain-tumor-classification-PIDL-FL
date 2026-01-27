"""
ResNet-18 model with feature map extraction for brain tumor classification.
Supports Perona-Malik regularization on intermediate feature maps.
Adapted for federated learning.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18FeatureExtractor(nn.Module):
    """
    ResNet-18 model that exposes intermediate feature maps (layer1, layer2, layer3, layer4).
    """
    def __init__(self, num_classes=4, pretrained=True):
        super(ResNet18FeatureExtractor, self).__init__()
        
        # Load pretrained ResNet-18
        # Handle deprecation: use weights parameter instead of pretrained
        if pretrained:
            try:
                # New API (torchvision >= 0.13)
                resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            except (AttributeError, TypeError):
                # Fallback for older torchvision versions
                resnet = models.resnet18(pretrained=pretrained)
        else:
            resnet = models.resnet18(weights=None)
        
        # Extract feature layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Global average pooling and classifier
        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)
        
        # Initialize classifier weights
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x, return_features=False):
        """
        Forward pass with optional feature map extraction.
        
        Args:
            x: Input tensor (B, C, H, W)
            return_features: If True, return intermediate feature maps
        
        Returns:
            If return_features=False: logits (B, num_classes)
            If return_features=True: (logits, feature_maps_dict)
            where feature_maps_dict contains 'layer1', 'layer2', 'layer3', 'layer4'
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        
        # Global average pooling
        x = self.avgpool(layer4)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        
        if return_features:
            feature_maps = {
                'layer1': layer1,
                'layer2': layer2,
                'layer3': layer3,
                'layer4': layer4
            }
            return logits, feature_maps
        
        return logits
