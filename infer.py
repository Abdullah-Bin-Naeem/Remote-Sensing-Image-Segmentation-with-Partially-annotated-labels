import os
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import tifffile as tiff
from model import ResNet18Segmentation

# Configuration
INPUT_DIR = "input_images"
OUTPUT_DIR = "output_images"
MODEL_PATH = r"C:\LD D\Academic\PARTIAL_SEG\checkpoints\best_model.pth"  # path to your trained model weights
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Color palette (same as training)
palette = {
    0: (255, 255, 255),  # impervious surfaces
    1: (0, 0, 255),      # buildings
    2: (0, 255, 255),    # low veg
    3: (0, 255, 0),      # trees
    4: (255, 255, 0),    # cars
    5: (255, 0, 0),      # clutter
    6: (0, 0, 0)         # undefined
}

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
num_classes = len(palette)
model = ResNet18Segmentation(num_classes=num_classes, output_size=(150, 150))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

# Preprocessing transform
def preprocess(image: Image.Image):
    # Resize to model input size
    image = image.resize((150, 150), Image.BILINEAR)
    # To tensor and normalize
    arr = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1xCxHxW
    return tensor

# Inference loop
for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith((".png", ".jpg", ".tif", ".tiff")):  # adjust as needed
        continue
    # Load image
    path = os.path.join(INPUT_DIR, fname)
    image = Image.open(path).convert("RGB")
    input_tensor = preprocess(image).to(DEVICE)

    # Forward pass
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()

    # Map to color mask
    h, w = pred.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_index, color in palette.items():
        mask = pred == cls_index
        color_mask[mask] = color

    # Save output
    out_name = os.path.splitext(fname)[0] + "_mask.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    mask_img = Image.fromarray(color_mask)
    mask_img.save(out_path)
    print(f"Saved segmentation mask to {out_path}")
