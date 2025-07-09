# ELEC 475 lab 4: Semantic Segmentation with Knowledge Distillation

A PyTorch implementation of knowledge distillation for semantic segmentation on the PASCAL VOC 2012 dataset. This project compares different distillation strategies to transfer knowledge from a heavy teacher model (FCN-ResNet50) to a lightweight student model (DeepLabV3-MobileNetV3).

## Project Overview

This project implements and evaluates three different training strategies for semantic segmentation, focusing on creating efficient lightweight models through knowledge distillation. The system trains a MobileNetV3-based student model to achieve comparable performance to a much larger ResNet50 teacher model while maintaining significantly faster inference speeds.

## Features

- **Lightweight Student Architecture**: DeepLabV3-MobileNetV3 optimized for speed and efficiency
- **Multiple Distillation Strategies**: Response-based and feature-based knowledge transfer
- **Comprehensive Evaluation Pipeline**: mIoU metrics, timing analysis, and visual comparisons
- **Automated Training Pipeline**: Class-weighted loss, learning rate scheduling, and checkpointing
- **Interactive Visualization**: Per-class analysis and prediction browsing tools
- **Cross-Platform Support**: Works with CUDA (GPU) and CPU

## Dataset Structure

The project uses PASCAL VOC 2012 segmentation dataset with automatic download:
```
├── data/                          # Auto-created dataset directory
│   └── VOCdevkit/
│       └── VOC2012/
│           ├── JPEGImages/        # RGB images
│           ├── SegmentationClass/ # Segmentation masks
│           └── ImageSets/         # Train/val splits
├── checkpoints/                   # Model outputs and plots
│   ├── no_distill_*/
│   ├── response_distill_*/
│   └── feature_distill_*/
└── ...
```

### Data Format
- **Input Images**: 256×256 RGB (resized from original)
- **Segmentation Masks**: 256×256 with 21 classes (0-20)
- **Classes**: 20 object classes + background
- **Train/Val Split**: ~1,464 training, ~1,449 validation images

## Model Architecture

**Teacher Model (FCN-ResNet50):**
- **Backbone**: ResNet50 (pretrained on ImageNet)
- **Head**: Fully Convolutional Network
- **Parameters**: ~35M
- **Purpose**: Provides rich feature representations for distillation

**Student Model (LightweightDeepLabV3):**
- **Backbone**: MobileNetV3-Large (pretrained)
- **Features**: 960 → 256 channel reduction, simplified ASPP
- **Parameters**: ~3.5M (10× smaller than teacher)
- **Architecture Details**:
  - Conv2d(960, 256, 1) + BatchNorm + ReLU
  - Conv2d(256, 256, 3, dilation=6) + BatchNorm + ReLU
  - Dropout(0.1)
  - Conv2d(256, 21, 1) for final classification

## Installation

### Requirements
```bash
pip install torch torchvision tqdm numpy matplotlib
```

### Dependencies
- **Python**: 3.7+
- **PyTorch**: 1.9+
- **torchvision**: Latest compatible version
- **Additional**: tqdm, numpy, matplotlib

### Dataset Setup
The PASCAL VOC 2012 dataset downloads automatically on first run. Ensure sufficient disk space (~2GB).

## Usage

### Training

**1. Baseline Training (No Distillation):**
```bash
python train_mobilenetv3_wce.py
```

**2. Response-based Knowledge Distillation:**
```bash
python response_distill_wce_fixplot.py
```

**3. Feature-based Knowledge Distillation:**
```bash
python feature_distill_wce_v4.py
```

**Training Parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `epochs` | 50 | Training epochs |
| `batch_size` | 16 | Batch size |
| `learning_rate` | 1e-4 | Initial learning rate |
| `T` | 2.0 | Temperature for response distillation |
| `soft_target_weight` | 0.5 | Weight for distillation loss |
| `ce_weight` | 0.5 | Weight for cross-entropy loss |

### Evaluation

**Model Performance Evaluation:**
```bash
# Baseline model
python evaluate-visualize-mobilenet.py --weights checkpoints/no_distill_*/best_model.pth

# Response distillation model
python evaluate-visualize-mobilenet.py --weights checkpoints/response_distill_*/best_response_model.pth

# Feature distillation model  
python evaluate-visualize-mobilenet.py --weights checkpoints/feature_distill_*/best_feature_model.pth
```

**Inference Timing Analysis:**
```bash
# Teacher model timing
python resnet50-miou-timed.py

# Student model timing
python mobilenet-miou-timed.py --weights [model_weights.pth]
```

**Evaluation Parameters:**

| Parameter | Description |
|-----------|-------------|
| `--weights` | Path to trained model weights (.pth file) |
| `--data-dir` | Custom dataset directory (default: ./data) |
| `--pretrained` | Use pretrained backbone (evaluation only) |

## Training Configuration

**Default Hyperparameters:**
- **Optimizer**: Adam (weight_decay=1e-5)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Loss Function**: Weighted CrossEntropyLoss (ignore_index=255)
- **Class Weights**: Automatically computed from training data
- **Image Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

**Knowledge Distillation Parameters:**
- **Temperature (T)**: 2.0 for probability softening
- **Feature Loss Weight**: 0.1 for intermediate features
- **Final Feature Weight**: 0.5 for output features
- **Similarity Metric**: Cosine embedding loss for feature distillation

## Expected Results

**Performance Comparison:**

| Model Type | mIoU (%) | Inference Time (ms) | FPS | Model Size |
|------------|----------|--------------------|----|------------|
| Teacher (FCN-ResNet50) | 65-70 | 50-80 | 12-20 | ~35M params |
| Student (Baseline) | 55-60 | 15-25 | 40-65 | ~3.5M params |
| Student (Response Distill.) | 58-63 | 15-25 | 40-65 | ~3.5M params |
| Student (Feature Distill.) | 60-65 | 15-25 | 40-65 | ~3.5M params |

**Output Files:**
- **Model Weights**: `best_model.pth`, `best_response_model.pth`, `best_feature_model.pth`
- **Training Plots**: Loss curves, mIoU progression, timing statistics
- **Checkpoints**: Every 5 epochs with full training state

## GPU Support

The code automatically detects and uses CUDA if available:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**GPU Memory Requirements:**
- **Training**: ~6-8GB VRAM (batch_size=16)
- **Evaluation**: ~2-4GB VRAM
- **Reduce batch_size** if encountering OOM errors

## File Structure

```
├── student_model.py                    # LightweightDeepLabV3 model definition
├── train_mobilenetv3_wce.py           # Baseline training script
├── response_distill_wce_fixplot.py    # Response-based distillation
├── feature_distill_wce_v4.py          # Feature-based distillation
├── evaluate-visualize-mobilenet.py    # Evaluation and visualization
├── mobilenet-miou-timed.py            # Student model timing
├── resnet50-miou-timed.py             # Teacher model timing
├── ConvNN.py                           # Alternative CNN architecture
├── train.txt                           # Training instructions
├── test.txt                            # Testing instructions
└── README.md                           # This file
```

## Troubleshooting

**Common Issues:**

1. **CUDA Out of Memory**: Reduce `batch_size` from 16 to 8 or 4 in training scripts
2. **Dataset Download Fails**: Check internet connection; manually download VOC2012 if needed
3. **Import Errors**: Ensure all dependencies installed: `pip install torch torchvision tqdm numpy matplotlib`
4. **Low mIoU Performance**: Verify class weights computation and data augmentation
5. **Slow Training**: Enable `torch.backends.cudnn.benchmark = True` for consistent input sizes

**Performance Tips:**
- Use `num_workers > 0` in DataLoader if sufficient CPU cores available
- Enable mixed precision training for faster training on modern GPUs
- Monitor GPU utilization to ensure optimal batch size

## Implementation Details

**Knowledge Distillation Strategies:**

1. **Response-based**: Distills soft targets from teacher's final predictions
   - Uses temperature scaling (T=2.0) to soften probability distributions
   - Combines KL divergence loss with standard cross-entropy

2. **Feature-based**: Distills intermediate feature representations
   - Matches student backbone features to teacher layer outputs
   - Uses cosine similarity loss with channel alignment layers
   - Combines intermediate and final feature distillation

**Data Augmentation:**
- Standard ImageNet normalization
- Random resize and crop during training
- Nearest neighbor interpolation for segmentation masks

## Acknowledgments

- [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) for the segmentation dataset
- PyTorch team for the deep learning framework
- Torchvision for pretrained models and dataset utilities