import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None, gamma=2.0, ignore_index=-1):
        super(PartialCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights  # Tensor of shape (num_classes,)
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets, mask):
        """
        Compute partial focal loss for point-annotated pixels with class weights
        Args:
            inputs: Tensor of shape (N, C, H, W) - model logits
            targets: Tensor of shape (N, H, W) - point labels (-1 for ignored pixels)
            mask: Tensor of shape (N, H, W) - 1 for annotated pixels, 0 otherwise
        """
        # Flatten tensors
        inputs = inputs.permute(0, 2, 3, 1).reshape(-1, inputs.size(1))  # (N*H*W, C)
        targets = targets.view(-1)  # (N*H*W)
        mask = mask.view(-1)  # (N*H*W)
        
        # Select annotated pixels
        valid = (mask == 1) & (targets != self.ignore_index)
        inputs_valid = inputs[valid]
        targets_valid = targets[valid]
        
        if inputs_valid.size(0) == 0:
            return torch.tensor(0.0, requires_grad=True, device=inputs.device)
        
        # Compute focal loss
        log_probs = F.log_softmax(inputs_valid, dim=-1)
        probs = torch.exp(log_probs)
        log_pt = log_probs[torch.arange(inputs_valid.size(0)), targets_valid]
        pt = probs[torch.arange(inputs_valid.size(0)), targets_valid]
        
        # Focal loss: - (1 - pt)^gamma * log(pt)
        focal_factor = (1 - pt).pow(self.gamma)
        focal_loss = -focal_factor * log_pt
        
        # Apply class weights, ensuring device compatibility
        if self.class_weights is not None:
            weights = self.class_weights.to(inputs.device)  # Move weights to the same device as inputs
            weights = weights[targets_valid]  # Index with valid targets
            focal_loss = focal_loss * weights
        
        # Sum and normalize by number of labeled pixels
        loss = focal_loss.sum() / valid.sum().float()
        return loss