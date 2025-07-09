# ELEC 475 Lab 2: Pet Nose Detection

A deep learning project for detecting and localizing pet noses in images using a Convolutional Neural Network (CNN). The model predicts the (x, y) coordinates of nose locations in cat and dog images.

## Project Overview

This project implements a regression-based approach to nose detection, where the model learns to predict the pixel coordinates of pet noses. The system supports various data augmentation techniques to improve model robustness and generalization.

## Features

- **CNN Architecture**: Custom convolutional neural network optimized for coordinate regression
- **Multiple Augmentation Strategies**: Support for flip, noise, rotation, and combinations
- **Flexible Training Pipeline**: Configurable training with different augmentation types
- **Comprehensive Evaluation**: Testing framework with distance metrics and visualization
- **Cross-Platform Support**: Works with CUDA (GPU) and CPU

## Dataset Structure

The project expects the following file structure:
```
├── images/                    # Directory containing pet images
├── train_noses.txt           # Training data (image_name, coordinates)
├── test_noses.txt            # Test data (image_name, coordinates)
└── ...
```

### Data Format
The annotation files (`train_noses.txt`, `test_noses.txt`) contain:
- **Column 1**: Image filename (e.g., `beagle_154.jpg`)
- **Column 2**: Nose coordinates in format `"(x, y)"` (e.g., `"(162, 169)"`)

## Model Architecture

**ConvNN Architecture:**
- **Input**: 227×227×3 RGB images
- **Conv Layer 1**: 64 filters, 3×3 kernel, stride 2
- **Conv Layer 2**: 128 filters, 3×3 kernel, stride 2  
- **Conv Layer 3**: 256 filters, 3×3 kernel, stride 2
- **Fully Connected**: 4096 → 1024 → 1024 → 2 (x, y coordinates)
- **Activation**: ReLU for all layers except output
- **Loss Function**: Mean Squared Error (MSE)

## Installation

### Requirements
```bash
pip install torch torchvision pandas matplotlib torchsummary
```

### Dataset Setup
1. Download the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
2. Place images in the `./images/` directory
3. Ensure annotation files (`train_noses.txt`, `test_noses.txt`) are in the root directory

## Usage

### Training

**Basic Training (No Augmentation):**
```bash
python train_val2.py -f "./images" -s "weights_basic.pth" -p "plot_basic.png" -a "u"
```

**Training with Augmentations:**
```bash
# Horizontal flip
python train_val2.py -f "./images" -s "weights_flip.pth" -p "plot_flip.png" -a "f"

# Gaussian noise
python train_val2.py -f "./images" -s "weights_noise.pth" -p "plot_noise.png" -a "n"

# Rotation (15° clockwise)
python train_val2.py -f "./images" -s "weights_rotate.pth" -p "plot_rotate.png" -a "r"

# Combined augmentations
python train_val2.py -f "./images" -s "weights_combined.pth" -p "plot_combined.png" -a "fnr"
```

**Training Parameters:**
- `-f`: Path to images folder
- `-s`: Output model weights file (.pth)
- `-p`: Output training plot file (.png)
- `-a`: Augmentation type (see table below)

| Augmentation Code | Description |
|-------------------|-------------|
| `u` | Unaugmented (original images only) |
| `f` | Horizontal flip |
| `n` | Gaussian noise |
| `r` | Rotation (15° clockwise) |
| `fn` | Flip + Noise |
| `fr` | Flip + Rotation |
| `nr` | Noise + Rotation |
| `fnr` | Flip + Noise + Rotation |

### Testing/Evaluation

```bash
# Test trained model
python test3.py -s "weights_model.pth" -f "./images"
```

**Output Metrics:**
- **Minimum Distance**: Best prediction accuracy
- **Maximum Distance**: Worst prediction accuracy  
- **Mean Distance**: Average Euclidean distance between predictions and ground truth
- **Standard Deviation**: Prediction consistency measure

The script also displays the best and worst prediction examples with visualizations.

## Augmentation Details

### 1. Horizontal Flip
- Mirrors images horizontally
- Adjusts nose coordinates: `new_x = image_width - original_x`

### 2. Gaussian Noise
- Adds random noise: `noise = 0.4 × image_std × random_normal`
- Clamps values to maintain valid pixel range

### 3. Rotation
- Rotates images 15° clockwise
- Transforms coordinates using rotation matrix:
  ```
  x_new = x_centered × cos(θ) + y_centered × sin(θ)
  y_new = -x_centered × sin(θ) + y_centered × cos(θ)
  ```

## Training Configuration

**Default Hyperparameters:**
- **Epochs**: 100
- **Batch Size**: 32
- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-5)
- **Scheduler**: ReduceLROnPlateau
- **Weight Initialization**: Xavier uniform



## Expected Results

The model learns to predict nose coordinates with varying accuracy depending on:
- **Image quality and clarity**
- **Pet pose and orientation** 
- **Lighting conditions**
- **Augmentation strategy used**

Distance metrics are measured in pixels on 227×227 images.

## GPU Support

The code automatically detects and uses CUDA if available:
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```



## Troubleshooting

**Common Issues:**

1. **CUDA Out of Memory**: Reduce batch size in `train_val2.py`
2. **Missing Images**: Ensure all images referenced in annotation files exist in the images folder
3. **Import Errors**: Verify all required packages are installed
4. **File Path Issues**: Use absolute paths if relative paths cause problems

## Acknowledgments

- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) for providing the pet images
- PyTorch team for the deep learning framework