# Remote Sensing Image Segmentation with Partial Cross Entropy Loss

This repository contains the implementation of a semantic segmentation model for remote sensing images using sparse point annotations, as described in the technical report titled "Partial Cross Entropy Loss for Remote Sensing Image Segmentation with Point Annotations." The model leverages a ResNet18-based architecture, a custom Partial Cross Entropy (CE) Loss function, and the ISPRS Potsdam dataset. The approach addresses class imbalance with frequency-based class weights and uses 20,000 point annotations per image (22.2% of the original 90,000 pixels in 300x300 images). An inference pipeline is provided to generate segmentation masks for new images.

## Project Overview

The project focuses on training a segmentation model with sparse point annotations, reducing the need for dense pixel-wise labels. Key components include:

- **Dataset**: ISPRS Potsdam dataset, with 2,100 training and 300 validation RGB images (300x300 pixels, resized to 150x150 for processing).
- **Model**: ResNet18 encoder (pretrained on ImageNet) with a decoder using transposed convolutional layers.
- **Loss Function**: Partial CE Loss with focal loss parameter (\(\gamma = 2\)) and class weights to handle imbalance.
- **Point Annotations**: 20,000 points sampled per image, covering approximately 22.2% of the original 90,000 pixels.
- **Experiments**: Two experiments compare performance with and without class weights, showing significant improvements with class weights (accuracy: 80.5% vs. 75.2%, mIoU: 65.7% vs. 58.3%).
- **Inference**: A pipeline to apply the trained model to new images, producing color-coded segmentation masks.

## Repository Structure

- `dataset.py`: Defines the `PotsdamDataset` class for loading and preprocessing the dataset, including point annotation simulation.
- `model.py`: Implements the `ResNet18Segmentation` model architecture.
- `partial_ce_loss.py`: Contains the `PartialCrossEntropyLoss` class for point-based training.
- `class_weights.py`: Computes class weights based on pixel frequencies.
- `train.ipynb`: Jupyter notebook for training the model and visualizing results.
- `infer.py`: Script for running inference on new images to generate segmentation masks.
- `archive/patches/`: Directory for storing dataset images and labels (not included in the repository).
- `checkpoints/`: Directory for saving trained model weights.
- `input_images/`: Directory for input images during inference.
- `output_images/`: Directory for saving generated segmentation masks.
- `images/output.png`: Training and validation curves (loss, accuracy, mIoU) for the experiment with class weights.
- `images/results.png`: Visual results showing input images, ground truth masks, point annotations, and predicted masks.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/remote-sensing-segmentation.git
   cd remote-sensing-segmentation
   ```

2. **Install Dependencies**:
   Ensure Python 3.8+ is installed. Install the required packages using:
   ```bash
   pip install torch torchvision numpy pillow tifffile scikit-image matplotlib
   ```

3. **Prepare the Dataset**:
   - Download the ISPRS Potsdam dataset from the [official source](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx).
   - Organize the dataset in `archive/patches/` with subdirectories `Images` (RGB images) and `Labels` (segmentation masks).
   - Expected file naming: `Image_{id}.tif` for images and `Label_{id}.tif` for labels, where `id` ranges from 0 to 2399.

4. **Download Pretrained Model** (optional):
   - Place the trained model weights (`best_model.pth`) in the `checkpoints/` directory for inference.
   - Alternatively, train the model using `train.ipynb` to generate your own weights.

5. **Add Result Images**:
   - Place `output.png` (training curves) and `results.png` (visual results) in the `images/` directory.
   - These images are generated during training (`train.ipynb`) or can be created manually to visualize results.

## Usage

### Training
1. Open `train.ipynb` in Jupyter Notebook or JupyterLab.
2. Update the `MAIN_FOLDER` path in the notebook to point to your dataset directory (`archive/patches`).
3. Run the notebook cells to:
   - Load the dataset.
   - Compute class weights (saved as `class_weights.pt`).
   - Train the model for 100 epochs with the Adam optimizer and StepLR scheduler.
   - Save the best model based on validation mIoU to `checkpoints/`.
   - Generate and save training curves (`output.png`) and visual results (`results.png`) to the `images/` directory.

### Inference
1. Place input images (PNG, JPG, TIF, or TIFF) in the `input_images/` directory.
2. Update `MODEL_PATH` in `infer.py` to point to your trained model weights (e.g., `checkpoints/best_model.pth`).
3. Run the inference script:
   ```bash
   python infer.py
   ```
4. Segmentation masks will be saved in `output_images/` with filenames like `original_name_mask.png`. Each mask uses the color palette:
   - Impervious surfaces: White (255, 255, 255)
   - Buildings: Blue (0, 0, 255)
   - Low vegetation: Cyan (0, 255, 255)
   - Trees: Green (0, 255, 0)
   - Cars: Yellow (255, 255, 0)
   - Clutter: Red (255, 0, 0)
   - Undefined: Black (0, 0, 0)

## Results

The model was evaluated on the ISPRS Potsdam validation set:
- **Without Class Weights**: Accuracy: 75.2%, mIoU: 58.3%
- **With Class Weights**: Accuracy: 80.5%, mIoU: 65.7%

The use of class weights significantly improves performance, especially for minority classes (e.g., cars, clutter). I hypothesize that increasing the number of point annotations beyond 20,000 would further enhance accuracy, though diminishing returns may occur as annotations approach full supervision.

### Training Curves
The following figure shows the training and validation loss, accuracy, and mIoU curves for the experiment with class weights, indicating stable convergence and good generalization.

![Training and Validation Curves](output.png)

### Visual Results
The figure below presents segmentation results for three validation samples, showing input images, ground truth masks, point annotations, and predicted masks. The predicted masks closely resemble the ground truth, with minor errors in cluttered regions.

![Visual Results](results.png)

## Future Work
- Experiment with varying numbers of point annotations to find an optimal balance between annotation effort and performance.
- Explore ensemble methods or test-time augmentation to boost segmentation accuracy.
- Extend the inference pipeline to support batch processing and additional image formats.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- ISPRS for providing the Potsdam dataset.
- The PyTorch community for tools and pretrained models.
