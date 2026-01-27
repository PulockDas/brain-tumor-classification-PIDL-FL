"""
Physics-informed regularizers for feature map regularization.
Includes Perona-Malik anisotropic diffusion and isotropic diffusion.
Adapted for federated learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_spatial_gradients(feature_map):
    """
    Compute spatial gradients using finite differences.
    
    Args:
        feature_map: Tensor of shape (B, C, H, W)
    
    Returns:
        grad_x: Gradient in x-direction (B, C, H, W)
        grad_y: Gradient in y-direction (B, C, H, W)
        grad_magnitude: |∇F| = sqrt(grad_x^2 + grad_y^2) (B, C, H, W)
    """
    # Pad to handle boundaries
    padded = F.pad(feature_map, (1, 1, 1, 1), mode='replicate')
    
    # Finite differences (central differences)
    # grad_x: difference in horizontal direction
    grad_x = (padded[:, :, :, 2:] - padded[:, :, :, :-2]) / 2.0
    grad_x = grad_x[:, :, 1:-1, :]  # Remove padding in y-direction
    
    # grad_y: difference in vertical direction
    grad_y = (padded[:, :, 2:, :] - padded[:, :, :-2, :]) / 2.0
    grad_y = grad_y[:, :, :, 1:-1]  # Remove padding in x-direction
    
    # Gradient magnitude
    grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
    
    return grad_x, grad_y, grad_magnitude


def perona_malik_loss(feature_map, k=1.0):
    """
    Compute Perona-Malik anisotropic diffusion regularization loss.
    
    The Perona-Malik equation uses a conduction function:
    c(s) = exp(-(s/k)^2)
    
    The physics loss is:
    L_pm = mean(c(|∇F|) * |∇F|^2)
    
    Args:
        feature_map: Tensor of shape (B, C, H, W)
        k: Conduction parameter (controls edge sensitivity)
    
    Returns:
        loss: Scalar tensor
    """
    _, _, grad_magnitude = compute_spatial_gradients(feature_map)
    
    # Conduction function: c(s) = exp(-(s/k)^2)
    s = grad_magnitude / k
    conduction = torch.exp(-(s ** 2))
    
    # Physics loss: mean(c(|∇F|) * |∇F|^2)
    loss = torch.mean(conduction * (grad_magnitude ** 2))
    
    return loss


def isotropic_diffusion_loss(feature_map):
    """
    Compute isotropic diffusion regularization loss (plain diffusion).
    
    L_iso = mean(|∇F|^2)
    
    Args:
        feature_map: Tensor of shape (B, C, H, W)
    
    Returns:
        loss: Scalar tensor
    """
    _, _, grad_magnitude = compute_spatial_gradients(feature_map)
    
    # Isotropic loss: mean(|∇F|^2)
    loss = torch.mean(grad_magnitude ** 2)
    
    return loss


class PIDLLoss(nn.Module):
    """
    Physics-Informed Deep Learning (PIDL) loss combining CE loss with physics regularizers.
    Supports Perona-Malik and isotropic diffusion.
    """
    def __init__(self, regularizer_type='perona_malik', k=1.0, lambda_pm=0.1, num_classes=4):
        """
        Args:
            regularizer_type: 'perona_malik', 'isotropic', or 'none'
            k: Conduction parameter for Perona-Malik (only used if regularizer_type='perona_malik')
            lambda_pm: Regularization weight
            num_classes: Number of classes
        """
        super(PIDLLoss, self).__init__()
        self.regularizer_type = regularizer_type
        self.k = k
        self.lambda_pm = lambda_pm
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, logits, labels, feature_map=None):
        """
        Compute total PIDL loss.
        
        Args:
            logits: Model predictions (B, num_classes)
            labels: Ground truth labels (B,)
            feature_map: Feature map for regularization (B, C, H, W) or None
        
        Returns:
            total_loss: Total loss (CE + physics regularization)
            ce_loss: Cross-entropy loss
            reg_loss: Regularization loss (0 if feature_map is None or lambda_pm=0)
        """
        # Classification loss
        ce_loss = self.ce_loss(logits, labels)
        
        # Regularization loss
        if feature_map is None or self.lambda_pm == 0.0 or self.regularizer_type == 'none':
            reg_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        elif self.regularizer_type == 'perona_malik':
            reg_loss = perona_malik_loss(feature_map, k=self.k)
            reg_loss = self.lambda_pm * reg_loss
        elif self.regularizer_type == 'isotropic':
            reg_loss = isotropic_diffusion_loss(feature_map)
            reg_loss = self.lambda_pm * reg_loss
        else:
            raise ValueError(f"Unknown regularizer_type: {self.regularizer_type}")
        
        total_loss = ce_loss + reg_loss
        
        return total_loss, ce_loss, reg_loss
