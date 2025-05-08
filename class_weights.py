import numpy as np
import torch
from torch.utils.data import DataLoader
import os

# Assuming PotsdamDataset is defined as in your previous code
from dataset import PotsdamDataset

def compute_class_weights(dataset, num_classes=7, undefined_class=6, smoothing=1e-6):
    """
    Compute class weights inversely proportional to class frequencies.
    
    Args:
        dataset: PotsdamDataset instance containing the data
        num_classes: Total number of classes (including undefined)
        undefined_class: Class index to exclude (undefined class)
        smoothing: Small value to avoid division by zero
    
    Returns:
        class_weights: Tensor of shape (num_classes,) with weights for each class
    """
    # Initialize frequency counts
    class_counts = np.zeros(num_classes, dtype=np.float32)
    
    # Create a DataLoader to iterate over the dataset
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Iterate over the dataset to count class occurrences
    for _, labels, _, _ in loader:  # We only need the full labels, not point labels
        labels = labels.numpy().flatten()
        for cls in range(num_classes):
            if cls == undefined_class:
                continue
            class_counts[cls] += np.sum(labels == cls)
    
    # Exclude undefined class by setting its count to a large value (weight will be small)
    class_counts[undefined_class] = np.inf
    
    # Add smoothing to avoid division by zero
    class_counts += smoothing
    
    # Compute inverse frequencies
    total_counts = np.sum(class_counts[class_counts != np.inf])
    class_frequencies = class_counts / total_counts
    class_weights = 1.0 / class_frequencies
    
    # Normalize weights so that the sum of weights (excluding undefined) equals the number of classes
    valid_mask = np.ones(num_classes, dtype=bool)
    valid_mask[undefined_class] = False
    class_weights[valid_mask] = class_weights[valid_mask] / np.sum(class_weights[valid_mask]) * np.sum(valid_mask)
    
    # Set weight for undefined class to 0
    class_weights[undefined_class] = 0.0
    
    return torch.tensor(class_weights, dtype=torch.float32)

# Example usage
if __name__ == "__main__":
    # Dataset parameters (adjust paths as needed)
    MAIN_FOLDER = "archive/patches"
    DATA_FOLDER = os.path.join(MAIN_FOLDER, 'Images')
    LABELS_FOLDER = os.path.join(MAIN_FOLDER, 'Labels')
    train_ids = list(range(0, 2100))  # Adjust based on your split
    
    # Create dataset
    train_dataset = PotsdamDataset(train_ids, DATA_FOLDER, LABELS_FOLDER, num_points=100)
    
    # Compute class weights
    class_weights = compute_class_weights(train_dataset)
    print("Class Weights:", class_weights)
    
    # Optionally save weights for later use
    torch.save(class_weights, "class_weights.pt")