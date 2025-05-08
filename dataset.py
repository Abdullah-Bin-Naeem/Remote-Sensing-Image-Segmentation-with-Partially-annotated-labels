# import numpy as np
# import torch
# import torch.utils.data as data
# from torch.utils.data import DataLoader
# import tifffile as tiff
# from skimage import io
# import os
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# import torch.nn.functional as F
# import os



# # Color palette for Potsdam dataset
# palette = {
#     0: (255, 255, 255),  # Impervious surfaces (white)
#     1: (0, 0, 255),     # Buildings (blue)
#     2: (0, 255, 255),   # Low vegetation (cyan)
#     3: (0, 255, 0),     # Trees (green)
#     4: (255, 255, 0),   # Cars (yellow)
#     5: (255, 0, 0),     # Clutter (red)
#     6: (0, 0, 0)        # Undefined (black)
# }
# invert_palette = {v: k for k, v in palette.items()}

# def convert_from_color(arr_3d, palette=invert_palette):
#     """Convert RGB color mask to grayscale labels"""
#     arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
#     for c, i in palette.items():
#         m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
#         arr_2d[m] = i
#     return arr_2d

# class PotsdamDataset(data.Dataset):
#     def __init__(self, ids, data_folder, labels_folder, target_size=(150, 150)):
#         super(PotsdamDataset, self).__init__()
#         self.target_size = target_size
        
#         # Initialize file paths
#         self.data_files = [os.path.join(data_folder, f'Image_{id}.tif') for id in ids]
#         self.label_files = [os.path.join(labels_folder, f'Label_{id}.tif') for id in ids]
        
#         # Validate files
#         self.valid_files = []
#         for data_f, label_f in zip(self.data_files, self.label_files):
#             if os.path.isfile(data_f) and os.path.isfile(label_f):
#                 self.valid_files.append((data_f, label_f))
#             else:
#                 print(f"Warning: Skipping missing file pair - Data: {data_f}, Label: {label_f}")
        
#         if not self.valid_files:
#             raise ValueError("No valid file pairs found in the specified directories")

#     def __len__(self):
#         return len(self.valid_files)

#     def __getitem__(self, idx):
#         data_path, label_path = self.valid_files[idx]
        
#         # Load and preprocess image
#         data = tiff.imread(data_path).transpose((2, 0, 1))  # To CHW format
#         data = data / 255.0  # Normalize to [0,1]
#         data = np.asarray(data, dtype=np.float32)
        
#         # Load and preprocess label
#         label = convert_from_color(tiff.imread(label_path))
#         label = np.asarray(label, dtype=np.int64)
        
#         # Resize if needed
#         if self.target_size:
#             data = torch.tensor(data).unsqueeze(0)
#             data = F.interpolate(data, size=self.target_size, mode='bilinear', align_corners=False)
#             data = data.squeeze(0).numpy()
            
#             label = torch.tensor(label).unsqueeze(0).unsqueeze(0)
#             label = F.interpolate(label.float(), size=self.target_size, mode='nearest')
#             label = label.squeeze(0).squeeze(0).long().numpy()

#         return torch.from_numpy(data), torch.from_numpy(label)

# def visualize_sample(dataset, model=None, device='cpu', num_samples=3):
#     """Visualize random samples with images, true masks, and predicted masks (if model provided)"""
#     # Create custom colormap from palette
#     colors = [np.array(v) / 255.0 for v in palette.values()]
#     cmap = ListedColormap(colors)
    
#     loader = DataLoader(dataset, batch_size=1, shuffle=True)
#     samples_shown = 0
    
#     if model:
#         model.eval()
    
#     with torch.no_grad():
#         for data, labels in loader:
#             if samples_shown >= num_samples:
#                 break
                
#             data, labels = data.to(device), labels.to(device)
            
#             # Get predictions if model is provided
#             predicted = None
#             if model:
#                 outputs = model(data)
#                 _, predicted = torch.max(outputs, 1)
            
#             # Convert to numpy for visualization
#             data_np = data.squeeze(0).permute(1, 2, 0).cpu().numpy()
#             true_mask_np = labels.squeeze(0).cpu().numpy()
#             predicted_mask_np = predicted.squeeze(0).cpu().numpy() if predicted is not None else None
            
#             # Plot
#             fig = plt.figure(figsize=(15, 4))
            
#             # Image
#             plt.subplot(1, 3 if model else 2, 1)
#             plt.imshow(data_np)
#             plt.title('Input Image')
#             plt.axis('off')
            
#             # True Mask
#             plt.subplot(1, 3 if model else 2, 2)
#             plt.imshow(true_mask_np, cmap=cmap, vmin=0, vmax=len(palette)-1)
#             plt.title('Ground Truth Mask')
#             plt.axis('off')
            
#             # Predicted Mask
#             if model:
#                 plt.subplot(1, 3, 3)
#                 plt.imshow(predicted_mask_np, cmap=cmap, vmin=0, vmax=len(palette)-1)
#                 plt.title('Predicted Mask')
#                 plt.axis('off')
            
#             plt.tight_layout()
#             plt.show()
            
#             samples_shown += 1 

# ----------------------------------------takes points randomly
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import tifffile as tiff
from skimage import io
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch.nn.functional as F

# Color palette for Potsdam dataset
palette = {
    0: (255, 255, 255),  # Impervious surfaces (white)
    1: (0, 0, 255),     # Buildings (blue)
    2: (0, 255, 255),   # Low vegetation (cyan)
    3: (0, 255, 0),     # Trees (green)
    4: (255, 255, 0),   # Cars (yellow)
    5: (255, 0, 0),     # Clutter (red)
    6: (0, 0, 0)        # Undefined (black)
}
invert_palette = {v: k for k, v in palette.items()}

def convert_from_color(arr_3d, palette=invert_palette):
    """Convert RGB color mask to grayscale labels"""
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i
    return arr_2d

def simulate_point_labels(label, num_points=100):
    """Simulate point-based annotations by randomly sampling points from the label"""
    h, w = label.shape
    point_label = np.full_like(label, fill_value=-1, dtype=np.int64)  # -1 for ignored pixels
    point_mask = np.zeros_like(label, dtype=np.float32)  # 0 for ignored, 1 for annotated
    
    # Flatten the label and get valid indices
    valid_indices = np.where(label.flatten() != 6)[0]  # Exclude undefined class (6)
    if len(valid_indices) == 0:
        return point_label, point_mask
    
    # Randomly sample points
    num_points = min(num_points, len(valid_indices))
    sampled_indices = np.random.choice(valid_indices, size=num_points, replace=False)
    
    # Convert flat indices to 2D coordinates
    coords = np.unravel_index(sampled_indices, (h, w))
    
    # Assign labels and mask
    point_label[coords] = label[coords]
    point_mask[coords] = 1.0
    
    return point_label, point_mask

class PotsdamDataset(data.Dataset):
    def __init__(self, ids, data_folder, labels_folder, target_size=(150, 150), num_points=100):
        super(PotsdamDataset, self).__init__()
        self.target_size = target_size
        self.num_points = num_points
        
        # Initialize file paths
        self.data_files = [os.path.join(data_folder, f'Image_{id}.tif') for id in ids]
        self.label_files = [os.path.join(labels_folder, f'Label_{id}.tif') for id in ids]
        
        # Validate files
        self.valid_files = []
        for data_f, label_f in zip(self.data_files, self.label_files):
            if os.path.isfile(data_f) and os.path.isfile(label_f):
                self.valid_files.append((data_f, label_f))
            else:
                print(f"Warning: Skipping missing file pair - Data: {data_f}, Label: {label_f}")
        
        if not self.valid_files:
            raise ValueError("No valid file pairs found in the specified directories")

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        data_path, label_path = self.valid_files[idx]
        
        # Load and preprocess image
        data = tiff.imread(data_path).transpose((2, 0, 1))  # To CHW format
        data = data / 255.0  # Normalize to [0,1]
        data = np.asarray(data, dtype=np.float32)
        
        # Load and preprocess label
        label = convert_from_color(tiff.imread(label_path))
        label = np.asarray(label, dtype=np.int64)
        
        # Simulate point labels
        point_label, point_mask = simulate_point_labels(label, num_points=self.num_points)
        
        # Resize if needed
        if self.target_size:
            data = torch.tensor(data).unsqueeze(0)
            data = F.interpolate(data, size=self.target_size, mode='bilinear', align_corners=False)
            data = data.squeeze(0).numpy()
            
            label = torch.tensor(label).unsqueeze(0).unsqueeze(0)
            label = F.interpolate(label.float(), size=self.target_size, mode='nearest')
            label = label.squeeze(0).squeeze(0).long().numpy()
            
            point_label = torch.tensor(point_label).unsqueeze(0).unsqueeze(0)
            point_label = F.interpolate(point_label.float(), size=self.target_size, mode='nearest')
            point_label = point_label.squeeze(0).squeeze(0).long().numpy()
            point_mask = torch.tensor(point_mask).unsqueeze(0).unsqueeze(0)
            point_mask = F.interpolate(point_mask, size=self.target_size, mode='nearest')
            point_mask = point_mask.squeeze(0).squeeze(0).numpy()

        return torch.from_numpy(data), torch.from_numpy(label), torch.from_numpy(point_label), torch.from_numpy(point_mask)

def visualize_sample(dataset, model=None, device='cpu', num_samples=3):
    """Visualize random samples with images, true masks, point masks, and predicted masks (if model provided)"""
    colors = [np.array(v) / 255.0 for v in palette.values()]
    cmap = ListedColormap(colors)
    
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    samples_shown = 0
    
    if model:
        model.eval()
    
    with torch.no_grad():
        for data, labels, point_labels, point_masks in loader:
            if samples_shown >= num_samples:
                break
                
            data, labels, point_labels, point_masks = data.to(device), labels.to(device), point_labels.to(device), point_masks.to(device)
            
            # Get predictions if model is provided
            predicted = None
            if model:
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
            
            # Convert to numpy for visualization
            data_np = data.squeeze(0).permute(1, 2, 0).cpu().numpy()
            true_mask_np = labels.squeeze(0).cpu().numpy()
            point_mask_np = point_labels.squeeze(0).cpu().numpy()
            predicted_mask_np = predicted.squeeze(0).cpu().numpy() if predicted is not None else None
            
            # Plot
            fig = plt.figure(figsize=(20, 4))
            
            # Image
            plt.subplot(1, 4 if model else 3, 1)
            plt.imshow(data_np)
            plt.title('Input Image')
            plt.axis('off')
            
            # True Mask
            plt.subplot(1, 4 if model else 3, 2)
            plt.imshow(true_mask_np, cmap=cmap, vmin=0, vmax=len(palette)-1)
            plt.title('Ground Truth Mask')
            plt.axis('off')
            
            # Point Mask
            plt.subplot(1, 4 if model else 3, 3)
            plt.imshow(np.where(point_mask_np >= 0, point_mask_np, np.nan), cmap=cmap, vmin=0, vmax=len(palette)-1)
            plt.title('Point Annotations')
            plt.axis('off')
            
            # Predicted Mask
            if model:
                plt.subplot(1, 4, 4)
                plt.imshow(predicted_mask_np, cmap=cmap, vmin=0, vmax=len(palette)-1)
                plt.title('Predicted Mask')
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
            samples_shown += 1
