# ELEC 475 Lab 1: MLP Autoencoder

A PyTorch implementation of a 4-layer Multi-Layer Perceptron (MLP) autoencoder for MNIST digit reconstruction, denoising, and interpolation.

## Overview

This project implements an autoencoder neural network that can:
- **Reconstruct** MNIST handwritten digits
- **Denoise** corrupted images  
- **Interpolate** between different digit images
- **Compress** images through a bottleneck layer

The autoencoder uses a symmetric 4-layer architecture: 784 → 392 → bottleneck → 392 → 784, where the bottleneck size is configurable.

## Project Structure

```
├── train.py          # Training script for the autoencoder
├── test.py           # Basic reconstruction testing (Step 1)
├── lab1.py           # Multi-step testing script (Steps 4-6)
├── model.py          # Autoencoder model definition
└── README.md         # This file
```

## Requirements

```bash
pip install torch torchvision matplotlib numpy torchsummary
```

## Model Architecture

The `autoencoderMLP4Layer` consists of:

**Encoder:**
- Linear: 784 → 392 (ReLU activation)
- Linear: 392 → bottleneck_size (ReLU activation)

**Decoder:**
- Linear: bottleneck_size → 392 (ReLU activation)  
- Linear: 392 → 784 (Sigmoid activation)

## Usage

### 1. Training the Model

Train an autoencoder with default parameters:

```bash
python train.py
```

Training with custom parameters:

```bash
python train.py -z 16 -e 50 -b 128 -s my_weights.pth -p my_plot.png
```

**Parameters:**
- `-z`: Bottleneck size (default: 32)
- `-e`: Number of epochs (default: 30)
- `-b`: Batch size (default: 256)
- `-s`: Save file for model weights (default: weights.pth)
- `-p`: Plot file for training loss (default: plot.png)

### 2. Basic Image Reconstruction (Step 1)

Test the trained model on individual MNIST images:

```bash
python test.py -s weights.pth -z 32
```

This will prompt you to enter image indices and display the original vs reconstructed images side by side.

### 3. Advanced Testing (Steps 4-6)

The `lab1.py` file contains three different testing modes. Uncomment the desired section:

#### Step 4: Image Reconstruction
```bash
python lab1.py -l weights.pth -z 32
```
Shows original and reconstructed images side by side.

#### Step 5: Denoising (Uncomment Step 5 section)
```bash
python lab1.py -l weights.pth -z 32
```
Adds random noise to images and shows: original → noisy → denoised

#### Step 6: Image Interpolation (Uncomment Step 6 section)
```bash
python lab1.py -l weights.pth -z 32
```
Interpolates between two images in the latent space, showing smooth transitions.

## Key Features

### Automatic Device Selection
The code automatically uses GPU if available, otherwise falls back to CPU.

### Xavier Weight Initialization
Weights are initialized using Xavier uniform distribution for better convergence.

### Learning Rate Scheduling
Uses ReduceLROnPlateau scheduler to adaptively reduce learning rate.

### Real-time Training Monitoring
- Displays training progress with timestamps
- Saves model weights after each epoch
- Generates loss plots automatically

## File Details

### `train.py`
- Main training loop with MSE loss
- Automatic model saving and loss plotting
- Command-line argument parsing
- Device-agnostic training (CPU/GPU)

### `model.py`
- Defines the `autoencoderMLP4Layer` class
- Separate `encode()` and `decode()` methods for flexibility
- Configurable bottleneck size

### `test.py` / `lab1.py`
- Interactive testing interfaces
- Image preprocessing and normalization
- Matplotlib visualization
- Support for different testing scenarios

## Example Results

The autoencoder can achieve good reconstruction quality on MNIST digits. Smaller bottleneck sizes (e.g., 8-16) create more compressed representations, while larger sizes (e.g., 32-64) preserve more detail.

**Typical compression ratios:**
- Bottleneck size 8: ~98% compression
- Bottleneck size 16: ~96% compression  
- Bottleneck size 32: ~92% compression

## Tips for Best Results

1. **Bottleneck Size**: Start with 32, reduce for more compression
2. **Training Time**: 20-30 epochs usually sufficient
3. **Batch Size**: 256 works well for most systems
4. **Learning Rate**: Default Adam optimizer settings work well

## Troubleshooting

**Common Issues:**

- **"File not found" error**: Make sure you've trained a model first with `train.py`
- **Memory issues**: Reduce batch size with `-b` parameter
- **Poor reconstruction**: Try increasing bottleneck size or training longer

## License

This project is part of ELEC 475 coursework and is intended for educational purposes.

## Acknowledgments

Based on ELEC 475 Lab 1 (Fall 2023) coursework for learning autoencoder fundamentals with PyTorch.