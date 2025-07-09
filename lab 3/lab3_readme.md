# ELEC 475 Lab 3: Ensemble Methods

A PyTorch implementation of transfer learning on CIFAR-100 dataset using pre-trained CNN architectures (AlexNet, VGG16, ResNet18) with ensemble evaluation methods.

## Overview

This project implements transfer learning by freezing the feature extraction layers of pre-trained models and fine-tuning only the classifier layers for CIFAR-100 classification. The project includes both individual model training/evaluation and ensemble methods comparison.

### Key Features

- **Transfer Learning**: Uses pre-trained AlexNet, VGG16, and ResNet18 models
- **Frozen Feature Extraction**: Only trains the final classification layers
- **Individual Model Evaluation**: Comprehensive metrics for each model
- **Ensemble Methods**: Three different ensemble approaches for improved performance
- **Automated Evaluation**: Batch evaluation of multiple model checkpoints



## Requirements

```bash
torch
torchvision
matplotlib
pandas
torchsummary
argparse
pathlib
datetime
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cifar100-transfer-learning
```

2. Install dependencies:
```bash
pip install torch torchvision matplotlib pandas torchsummary
```

3. The CIFAR-100 dataset will be automatically downloaded on first run.

## Usage

### Training Models

Train individual models using the provided training scripts:

```bash
# Train AlexNet
python train_freeze_.py -m alexnet -s alexnet_lr1e4.pth -p alexnet.png

# Train ResNet18
python train_freeze_.py -m resnet18 -s resnet18_lr1e4.pth -p resnet18.png

# Train VGG16
python train_freeze_.py -m vgg16 -s vgg16_lr1e4.pth -p vgg16.png
```

#### Training Parameters

- `-m`: Model architecture (alexnet, vgg16, resnet18)
- `-s`: Save file path for model weights (.pth)
- `-p`: Plot file path for training curves (.png)
- `-f`: Path to image folder (optional)

### Model Evaluation

#### Individual Model Evaluation

Evaluate all models in a directory:

```bash
python model-evaluation-dec.py --weights_dir ./ --output_file output.csv --batch_size 64
```

**Parameters:**
- `--weights_dir`: Directory containing .pth weight files
- `--output_file`: CSV file to save results (default: evaluation_results.csv)
- `--batch_size`: Batch size for evaluation (default: 64)

#### Ensemble Evaluation

Compare different ensemble methods:

```bash
# Ensemble with 5-epoch models
python ensemble-top1.py --alexnet_weights 5_alexnet_lr1e4.pth --resnet18_weights 5_resnet18_lr1e4.pth --vgg16_weights 5_vgg16_lr1e4.pth --output_file ensemble_5.csv

# Ensemble with fully trained models
python ensemble-top1.py --alexnet_weights alexnet_lr1e4.pth --resnet18_weights resnet18_lr1e4.pth --vgg16_weights vgg16_lr1e4.pth --output_file ensemble_full.csv
```

## Model Architecture Details

### Transfer Learning Approach

All models use the same transfer learning strategy:

1. **Load pre-trained weights** from ImageNet
2. **Freeze all feature layers** (set `requires_grad=False`)
3. **Replace final classifier** for 100 CIFAR-100 classes
4. **Train only the classifier** with Adam optimizer

### Supported Models

- **AlexNet**: Replaces `classifier[6]` with Linear(4096, 100)
- **VGG16**: Replaces `classifier[6]` with Linear(4096, 100)  
- **ResNet18**: Replaces `fc` with Linear(512, 100)

### Training Configuration

- **Optimizer**: Adam (lr=1e-4, weight_decay=1e-5)
- **Scheduler**: ReduceLROnPlateau (factor=0.1, patience=10)
- **Loss Function**: CrossEntropyLoss
- **Epochs**: 50 (with intermediate saves at epoch 5)
- **Batch Size**: 64
- **Image Size**: 224×224 (resized from 32×32)

## Ensemble Methods

The project implements three ensemble approaches:

1. **Maximum Probability**: Takes the maximum probability across models for each class
2. **Probability Averaging**: Averages the softmax probabilities from all models
3. **Majority Voting**: Uses the most frequently predicted class across models

## Evaluation Metrics

Both individual and ensemble evaluations provide:

- **Top-1 Accuracy**: Percentage of correct top predictions
- **Top-5 Accuracy**: Percentage where correct label is in top 5 predictions
- **Top-1 Error Rate**: 1 - Top-1 Accuracy
- **Top-5 Error Rate**: 1 - Top-5 Accuracy

## Data Preprocessing

CIFAR-100 images are preprocessed with:

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                        std=[0.2675, 0.2565, 0.2761])
])
```

## Output Files

### Training Outputs
- **Model weights**: `.pth` files for each epoch and final model
- **Training plots**: Loss curves saved as `.png` files

### Evaluation Outputs
- **Individual results**: CSV with metrics for each model
- **Ensemble results**: CSV comparing ensemble methods

### Sample CSV Output Format

**Individual Evaluation:**
```csv
filename,model_type,epoch,top1_error,top5_error,top1_accuracy,top5_accuracy,evaluation_time
alexnet_lr1e4.pth,alexnet,final,0.2134,0.0567,0.7866,0.9433,2024-01-15 10:30:45
```

**Ensemble Evaluation:**
```csv
ensemble_method,error_rate,accuracy
max_probability,0.1892,0.8108
probability_averaging,0.1845,0.8155
majority_voting,0.1923,0.8077
```

## Performance Tips

- Use GPU acceleration when available (automatically detected)
- Adjust batch size based on available memory
- Monitor training plots to detect overfitting
- Compare ensemble methods to find the best approach for your use case

## Acknowledgments

This project builds upon several key resources and frameworks:

- **PyTorch and Torchvision**: This implementation uses pre-trained models and datasets from PyTorch Vision. 
  - Models: https://docs.pytorch.org/vision/main/models.html
  - CIFAR-100 Dataset: https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR100.html

- **Pre-trained Models**: AlexNet, VGG16, and ResNet18 architectures with ImageNet pre-trained weights from torchvision.models

- **CIFAR-100 Dataset**: The 100-class image classification dataset by Krizhevsky & Hinton (2009), accessed through torchvision.datasets